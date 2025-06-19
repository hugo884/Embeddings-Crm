import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class Settings:
    # Configuración del modelo
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    MODEL_DEVICE: str = os.getenv("MODEL_DEVICE", "cpu")
    
    # Configuración de la caché
    CACHE_MAX_SIZE: int = int(os.getenv("CACHE_MAX_SIZE", "10000"))
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "7200"))  # en segundos
    
    # Configuración del thread pool
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    
    # Configuración de seguridad
    API_KEYS: list = os.getenv("API_KEYS", "").split(",")
    
    # Configuración del servidor
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    RELOAD: bool = os.getenv("RELOAD", "false").lower() == "true"

settings = Settings()