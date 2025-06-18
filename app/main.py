from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .controllers.embedding_controller import router as embed_router
from .dependencies import init_components

app = FastAPI()

# CORS: para permitir solicitudes de cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir las rutas definidas en el controlador de embedding
app.include_router(embed_router)

# Inicialización de modelo, caché y executor al arrancar el servidor
@app.on_event("startup")
async def startup():
    await init_components()
