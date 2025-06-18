from dotenv import load_dotenv
import os
import math
import psutil
import torch
import logging
from concurrent.futures import ThreadPoolExecutor
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from .model_loader import EmbeddingModel
from .utils.cache_utils import EmbeddingCache
from .services.embedding_service import EmbeddingService

# Cargar archivo .env
load_dotenv()

# Configuración de seguridad
API_KEYS = os.getenv("API_KEYS", "").split(",")
api_key_header = APIKeyHeader(name="X-API-KEY")

def validate_api_key(api_key: str = Security(api_key_header)):
    if api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

# Configuración de componentes globales
model: EmbeddingModel | None = None
cache: EmbeddingCache | None = None
executor: ThreadPoolExecutor | None = None

# Función para ajustar dinámicamente el número de hilos según la memoria disponible
def calculate_threads():
    free_memory = psutil.virtual_memory().available
    # Establecer un número de hilos dependiendo de la memoria disponible
    max_threads = min(os.cpu_count(), free_memory // (1024 ** 3))  # Aproximadamente 1 hilo por GB de RAM
    return max(2, max_threads)

# Función para inicializar los componentes (modelo, caché y executor)
async def init_components():
    global model, cache, executor

    total_ram = psutil.virtual_memory().total
    cpu_count = os.cpu_count() or 1
    max_workers = max(2, min(cpu_count, 8))
    cache_size = min(20_000, math.floor(total_ram / (768 * 4 * 1.5)))

    torch.set_num_threads(max(1, min(cpu_count // 2, 4)))  # Ajuste del número de hilos de PyTorch

    torch_config = {
        "num_threads": torch.get_num_threads(),
        "quantize": total_ram < 12 * 1024**3
    }

    logger = logging.getLogger(__name__)
    logger.info("⚙️ Inicializando modelo y caché...")

    model = EmbeddingModel(torch_config)
    cache = EmbeddingCache(max_size=cache_size, ttl=7200)
    executor = ThreadPoolExecutor(max_workers=max_workers)

    logger.info("✅ Componentes listos")

# Función para obtener el servicio de embeddings
def get_embedding_service() -> EmbeddingService:
    if not (model and cache and executor):
        raise RuntimeError("Service not initialized")
    return EmbeddingService(model, cache, executor)

# Obtener estado completo de los componentes
def get_status():
    return {
        "model_initialized": model is not None,
        "cache_initialized": cache is not None,
        "executor_initialized": executor is not None,
        "max_workers": executor._max_workers if executor else 0,
        # USAR LAS NUEVAS FUNCIONES DE LA CACHÉ
        "cache_size": cache.get_size() if cache else 0,
        "cache_usage": cache.get_usage() if cache else 0
    }