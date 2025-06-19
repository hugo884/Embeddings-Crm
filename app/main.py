from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .controllers.embedding_controller import router as embed_router
from .dependencies import init_components
import uvicorn
import logging

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Embedding Service API",
    description="Servicio optimizado para generación de embeddings",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS: para permitir solicitudes de cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir las rutas definidas en el controlador de embedding
app.include_router(embed_router, prefix="/api")

# Inicialización de componentes al arrancar el servidor
@app.on_event("startup")
def startup_event():
    logger.info("🚀 Iniciando servicio de embeddings...")
    try:
        init_components()
        logger.info("✅ Servicio inicializado correctamente")
    except Exception as e:
        logger.critical(f"Fallo en inicialización: {str(e)}")
        # Forzar cierre si no se pudo inicializar
        raise SystemExit(1)

# Evento de apagado para liberar recursos
@app.on_event("shutdown")
def shutdown_event():
    logger.info("🛑 Apagando servicio de embeddings...")
    # Aquí podrías agregar lógica para liberar recursos si es necesario

# Ruta raíz adicional
@app.get("/")
def root():
    return {
        "service": "embedding-service",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )