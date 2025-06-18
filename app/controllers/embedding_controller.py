from fastapi import APIRouter, Security, Depends
import psutil
import torch
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
    status = get_status()  # Obtener el estado de los componentes

    # Verificar si los componentes están inicializados
    if not status["model_initialized"]:
        return {"status": "INITIALIZING"}

    mem = psutil.virtual_memory()

    # Devolver el estado si todo está inicializado
    return {
        "status": "OK",
        "model_dimensions": get_embedding_service().model.model.get_sentence_embedding_dimension(),
        "torch_threads": torch.get_num_threads(),
        "max_workers": status["max_workers"],
        # Usar el tamaño de caché del estado
        "cache_size": status["cache_size"],
        "system_ram": f"{mem.used/1024**3:.1f}/{mem.total/1024**3:.1f} GB",
        "cpu_usage": f"{psutil.cpu_percent()}%"
    }

@router.get("/")
def home():
    return {"message": "Auto-Optimized Embedding Service for Electrical Products"}