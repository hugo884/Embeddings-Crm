from fastapi import APIRouter, Security, Depends
import psutil
import torch
import os
from ..schemas import EmbedRequest, EmbedResponse
from ..services.embedding_service import EmbeddingService
from ..dependencies import get_embedding_service, get_status, validate_api_key

router = APIRouter(tags=["Embedding"])

@router.post("/v1/embed", response_model=EmbedResponse)
async def embed_endpoint(
    body: EmbedRequest,
    api_key: str = Security(validate_api_key),
    service: EmbeddingService = Depends(get_embedding_service)
):
    embeddings, cached, total_time, batch_size = await service.embed_texts(body.texts)
    return {
        "embeddings": embeddings,
        "cached": cached,
        "processing_time": total_time,
        "batch_size": batch_size
    }

@router.get("/health")
def health_check():
    status = get_status()

    if not status["model_initialized"]:
        return {"status": "INITIALIZING"}

    mem = psutil.virtual_memory()
    service = get_embedding_service()

    # CORREGIR CÁLCULO DE CACHE_HIT_RATE
    cache_hit_rate = "N/A"
    if status['cache_size'] > 0:
        try:
            hit_rate = (status['cache_usage'] / status['cache_size']) * 100
            cache_hit_rate = f"{hit_rate:.1f}%"
        except ZeroDivisionError:
            cache_hit_rate = "0.0%"

    return {
        "status": "OK",
        "model": os.getenv("EMBEDDING_MODEL", "unknown"),
        "model_dimensions": service.model.get_dimensions(),
        "torch_threads": torch.get_num_threads(),
        "max_workers": status["max_workers"],
        "cache_size": status["cache_size"],
        "cache_usage": status["cache_usage"],
        "cache_hit_rate": cache_hit_rate,
        "system_ram": f"{mem.used/1024**3:.1f}/{mem.total/1024**3:.1f} GB",
        "cpu_usage": f"{psutil.cpu_percent()}%"
    }
    
    
@router.get("/")
def home():
    return {"message": "Auto-Optimized Embedding Service for Electrical Products"}