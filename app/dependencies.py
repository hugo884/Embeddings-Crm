from dotenv import load_dotenv
import os
import math
import psutil
import torch
import logging
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader
from .app_state import app_state

# Cargar archivo .env
load_dotenv()

# Configuración de seguridad
API_KEYS = os.getenv("API_KEYS", "").split(",")
api_key_header = APIKeyHeader(name="X-API-KEY")

logger = logging.getLogger(__name__)

def validate_api_key(api_key: str = Security(api_key_header)):
    """Valida la clave API proporcionada en el header"""
    if not API_KEYS or not API_KEYS[0]:
        logger.warning("No se han configurado API KEYS, permitiendo todas las solicitudes")
        return api_key
        
    if api_key not in API_KEYS:
        logger.warning(f"Intento de acceso con API key inválida: {api_key}")
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

def calculate_threads():
    """Calcula el número óptimo de hilos basado en recursos del sistema"""
    try:
        free_memory = psutil.virtual_memory().available
        # Establecer un número de hilos dependiendo de la memoria disponible
        max_threads = min(os.cpu_count() or 4, free_memory // (1024 ** 3))  # 1 hilo por GB de RAM
        return max(2, max_threads)
    except Exception as e:
        logger.error(f"Error calculando hilos: {e}", exc_info=True)
        return 4  # Valor por defecto

def init_components():
    """Inicializa todos los componentes principales del servicio"""
    try:
        logger.info("⚙️ Inicializando componentes del servicio...")
        
        # Obtener estadísticas del sistema
        total_ram = psutil.virtual_memory().total
        cpu_count = os.cpu_count() or 4
        max_workers = calculate_threads()
        cache_size = min(20000, math.floor(total_ram / (768 * 4 * 1.5)))
        
        # Configurar PyTorch para optimizar rendimiento
        torch_threads = max(1, min(cpu_count // 2, 4))
        torch.set_num_threads(torch_threads)
        
        # Inicializar modelo
        from .model_loader import EmbeddingModel
        model = EmbeddingModel({
            "num_threads": torch_threads,
            "quantize": total_ram < 12 * 1024**3
        })
        logger.info(f"🧠 Modelo de embeddings inicializado | Hilos: {torch_threads}")
        
        # Inicializar caché
        from .utils.cache_utils import EmbeddingCache
        cache = EmbeddingCache(max_size=cache_size, ttl=7200)
        logger.info(f"💾 Caché inicializada | Tamaño: {cache_size} items")
        
        # Inicializar executor
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"🧵 ThreadPool creado | Workers: {max_workers}")
        
        # Inicializar servicio
        from .services.embedding_service import EmbeddingService
        embedding_service = EmbeddingService(
            model=model,
            cache=cache,
            executor=executor
        )
        logger.info("✅ Todos los componentes inicializados")
        
        # Asignar al app_state
        app_state.model = model
        app_state.cache = cache
        app_state.executor = executor
        app_state.embedding_service = embedding_service
        
    except Exception as e:
        logger.critical(f"Fallo crítico al inicializar componentes: {str(e)}", exc_info=True)
        raise RuntimeError(f"No se pudieron inicializar los componentes: {str(e)}")

def get_embedding_service():
    """Obtiene el servicio de embeddings, inicializando si es necesario"""
    if not app_state.embedding_service:
        logger.warning("EmbeddingService no inicializado, inicializando ahora")
        init_components()
    return app_state.embedding_service

def get_cache():
    """Obtiene la instancia de la caché"""
    if not app_state.cache:
        init_components()
    return app_state.cache

def get_model():
    """Obtiene el modelo de embeddings"""
    if not app_state.model:
        init_components()
    return app_state.model

def get_executor():
    """Obtiene el executor de hilos"""
    if not app_state.executor:
        init_components()
    return app_state.executor

def get_status() -> dict:
    """Devuelve el estado actual del servicio"""
    try:
        # Si no hay instancia de modelo, no está inicializado
        if not app_state.model:
            return {
                "model_initialized": False,
                "max_workers": 0,
                "cache_size": 0,
                "cache_usage": 0,
                "cache_hits": 0,
                "cache_misses": 0,
            }
        
        # Obtener estadísticas de la caché
        cache_stats = app_state.cache.get_stats() if app_state.cache else {}
        
        return {
            "model_initialized": app_state.model.is_loaded() if hasattr(app_state.model, 'is_loaded') else True,
            "max_workers": app_state.executor._max_workers if app_state.executor else 0,
            "cache_size": cache_stats.get("max_size", 0),
            "cache_usage": cache_stats.get("current_size", 0),
            "cache_hits": cache_stats.get("hits", 0),
            "cache_misses": cache_stats.get("misses", 0),
        }
    except Exception as e:
        logger.error(f"Error obteniendo estado: {str(e)}")
        return {
            "model_initialized": False,
            "max_workers": 0,
            "cache_size": 0,
            "cache_usage": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }