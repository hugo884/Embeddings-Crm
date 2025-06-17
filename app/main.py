from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from .model_loader import EmbeddingModel
from .cache_utils import EmbeddingCache
import numpy as np
import os
from dotenv import load_dotenv

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
model = EmbeddingModel()
cache = EmbeddingCache(max_size=5000, ttl=7200)
api_key_header = APIKeyHeader(name="X-API-KEY")

def validate_api_key(api_key: str = Security(api_key_header)):
    if api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

@app.post("/v1/embed")
async def generate_embedding(request: Request, texts: list[str], api_key: str = Security(validate_api_key)):
    print(f"📥 Received request for {len(texts)} texts")
    
    try:
        embeddings = []
        uncached_texts = []
        
        for text in texts:
            cached = cache.get(text)
            if cached is not None:
                embeddings.append(cached.tolist())
            else:
                uncached_texts.append(text)
                embeddings.append(None)
        
        if uncached_texts:
            print(f"🔍 Generating embeddings for {len(uncached_texts)} uncached texts")
            uncached_embeddings = model.embed(uncached_texts)
            
            idx = 0
            for i, emb in enumerate(embeddings):
                if emb is None:
                    embedding_array = uncached_embeddings[idx]
                    embeddings[i] = embedding_array.tolist()
                    cache.set(texts[i], embedding_array)
                    idx += 1
        
        return {"embeddings": embeddings, "cached": len(texts) - len(uncached_texts)}
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding generation error: {str(e)}"
        )

@app.get("/health")
def health_check():
    return {"status": "OK", "model_dimensions": model.model.get_sentence_embedding_dimension()}

@app.get("/")
def home():
    return {"message": "Embedding Service for Electrical Products"}