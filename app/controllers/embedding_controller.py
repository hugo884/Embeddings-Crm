from fastapi import APIRouter, Security, Depends, HTTPException
import psutil
import torch
import os
import logging
from ..schemas import EmbedItemRequest, EmbedItemResponse, EmbedRequest, EmbedResponse
from ..dependencies import get_embedding_service, get_status, validate_api_key

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/v1/embed", response_model=EmbedResponse, tags=["Embedding"])
async def embed_endpoint(
    body: EmbedRequest,
    api_key: str = Security(validate_api_key),
    service = Depends(get_embedding_service)
):
    try:
        # Validar entrada
        if not body.items or len(body.items) == 0:
            raise HTTPException(status_code=400, detail="Lista de items vacía")
        
        # Extraer textos manteniendo relación con IDs
        texts = [item.text for item in body.items]
        
        # Generar embeddings
        embeddings, cached, total_time, processed = await service.embed_texts(texts)
        
        # Construir respuesta
        response_items = [
            EmbedItemResponse(
                productId=body.items[i].productId,
                embedding=emb
            )
            for i, emb in enumerate(embeddings)
        ]
        
        return EmbedResponse(
            items=response_items,
            cached=cached,
            processing_time=total_time,
            batch_size=processed
        )
    except Exception as e:
        logger.error(f"Error en endpoint /embed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error procesando embeddings: {str(e)}")

@router.get("/health", tags=["System"])
def health_check():
    try:
        status = get_status()
        service = get_embedding_service()

        # Si el modelo no está inicializado
        if not status["model_initialized"]:
            return {"status": "INITIALIZING"}

        # Obtener métricas del sistema
        mem = psutil.virtual_memory()
        
        # Calcular tasa de aciertos de caché
        cache_hit_rate = "0.0%"
        total_requests = status["cache_hits"] + status["cache_misses"]
        if total_requests > 0:
            hit_rate = (status["cache_hits"] / total_requests) * 100
            cache_hit_rate = f"{hit_rate:.1f}%"

        return {
            "status": "OK",
            "model": os.getenv("EMBEDDING_MODEL", "unknown"),
            "model_dimensions": service.dims,
            "torch_threads": torch.get_num_threads(),
            "max_workers": status["max_workers"],
            "cache": {
                "max_size": status["cache_size"],
                "current_size": status["cache_usage"],
                "hits": status["cache_hits"],
                "misses": status["cache_misses"],
                "hit_rate": cache_hit_rate
            },
            "system_ram": f"{mem.used/1024**3:.1f}/{mem.total/1024**3:.1f} GB",
            "cpu_usage": f"{psutil.cpu_percent()}%"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return {
            "status": "ERROR",
            "message": str(e)
        }
    
@router.get("/", tags=["System"])
def home():
    return {"message": "Auto-Optimized Embedding Service for Electrical Products"}