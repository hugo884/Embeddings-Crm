import sys
from pathlib import Path

# Añadir directorio padre al path para resolver imports
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from app.model_loader import EmbeddingModel
from app.cache_utils import EmbeddingCache
import numpy as np
import os
import psutil
import math
import logging
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import torch

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Configuración seguridad
API_KEYS = os.getenv("API_KEYS", "").split(",")
model = None
cache = None
executor = None
api_key_header = APIKeyHeader(name="X-API-KEY")

def validate_api_key(api_key: str = Security(api_key_header)):
    if api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

@app.on_event("startup")
async def initialize_service():
    global model, cache, executor
    
    # Obtener recursos del sistema
    total_ram = psutil.virtual_memory().total
    cpu_count = os.cpu_count()
    
    logger.info(f"🚀 Recursos detectados: RAM={total_ram/1024**3:.2f}GB, CPUs={cpu_count}")
    
    # Calcular parámetros óptimos
    max_workers = max(2, min(cpu_count, 8))
    cache_size = min(20000, math.floor(total_ram / (768 * 4 * 1.5)))
    
    torch_config = {
        "num_threads": max(1, min(cpu_count // 2, 4)),
        "quantize": total_ram < 12 * 1024**3
    }
    
    logger.info(f"⚙️ Configuración óptima: Workers={max_workers}, Cache={cache_size}, Threads={torch_config['num_threads']}")
    
    # Configurar PyTorch
    torch.set_num_threads(torch_config["num_threads"])
    
    # Inicializar componentes
    model = EmbeddingModel(torch_config)
    cache = EmbeddingCache(max_size=cache_size, ttl=7200)
    executor = ThreadPoolExecutor(max_workers=max_workers)

@app.post("/v1/embed")
async def generate_embedding(request: Request, texts: list[str], api_key: str = Security(validate_api_key)):
    if not model or not cache:
        raise HTTPException(status_code=503, detail="Service initializing")
    
    start_time = time.time()
    total_texts = len(texts)
    
    try:
        # Fase 1: Verificar caché
        embeddings = []
        uncached_texts = []
        cache_map = {}
        
        for text in texts:
            cached = cache.get(text)
            if cached is not None:
                embeddings.append(cached.tolist())
            else:
                uncached_texts.append(text)
                embeddings.append(None)
                cache_map[text] = len(embeddings) - 1
        
        cache_time = time.time()
        cache_duration = cache_time - start_time
        
        # Fase 2: Procesamiento de textos no cacheados
        if uncached_texts:
            logger.info(f"🔍 Procesando {len(uncached_texts)} textos no cacheados")
            
            # Calcular batch size dinámico basado en recursos
            max_batch = min(64, max(8, math.floor(psutil.virtual_memory().available / (768 * 4 * 1500))))
            batch_size = min(max_batch, len(uncached_texts))
            batches = [uncached_texts[i:i+batch_size] 
                        for i in range(0, len(uncached_texts), batch_size)]
            
            # Procesar en paralelo
            loop = asyncio.get_running_loop()
            batch_tasks = []
            
            for batch in batches:
                future = loop.run_in_executor(
                    executor, 
                    lambda b=batch: model.encode(
                        b, 
                        batch_size=len(b),
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False
                    )
                )
                batch_tasks.append(future)
            
            # Esperar todos los resultados
            batch_embeddings = await asyncio.gather(*batch_tasks)
            
            # Combinar resultados
            uncached_embeddings = np.vstack(batch_embeddings)
            
            # Actualizar caché y resultados
            for i, text in enumerate(uncached_texts):
                idx = cache_map[text]
                embedding_array = uncached_embeddings[i]
                embeddings[idx] = embedding_array.tolist()
                cache.set(text, embedding_array)
        
        proc_time = time.time()
        proc_duration = proc_time - cache_time
        total_duration = proc_time - start_time
        
        logger.info(f"⏱️ Tiempos: Cache={cache_duration:.2f}s, "
                    f"Proc={proc_duration:.2f}s, Total={total_duration:.2f}s, "
                    f"Textos={total_texts}, CacheHit={total_texts - len(uncached_texts)}")
        
        return {
            "embeddings": embeddings,
            "cached": total_texts - len(uncached_texts),
            "processing_time": total_duration,
            "batch_size": len(uncached_texts) if uncached_texts else 0
        }
    
    except Exception as e:
        logger.error(f"Error generando embeddings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Embedding generation error: {str(e)}"
        )

@app.get("/health")
def health_check():
    if not model:
        return {"status": "INITIALIZING"}
    
    mem = psutil.virtual_memory()
    return {
        "status": "OK",
        "model_dimensions": model.model.get_sentence_embedding_dimension(),
        "torch_threads": torch.get_num_threads(),
        "max_workers": executor._max_workers if executor else 0,
        "cache_size": cache.max_size if cache else 0,
        "system_ram": f"{mem.used/1024**3:.1f}/{mem.total/1024**3:.1f} GB",
        "cpu_usage": f"{psutil.cpu_percent()}%"
    }

@app.get("/")
def home():
    return {"message": "Auto-Optimized Embedding Service for Electrical Products"}