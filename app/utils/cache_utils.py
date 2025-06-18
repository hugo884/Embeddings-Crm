from cachetools import TTLCache
import hashlib
from typing import List, Optional
import numpy as np

class EmbeddingCache:
    def __init__(self, max_size=10000, ttl=3600):
        self.cache = TTLCache(maxsize=max_size, ttl=ttl)
    
    def generate_key(self, text: str) -> str:
        """Genera clave única para el texto normalizado"""
        normalized = text.strip().lower()
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def get_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Obtiene embeddings para un lote de textos"""
        keys = [self.generate_key(text) for text in texts]
        return [self.cache.get(key) for key in keys]
    
    def set_batch(self, texts: List[str], embeddings: List[List[float]]):
        """Almacena embeddings para un lote de textos"""
        keys = [self.generate_key(text) for text in texts]
        for key, emb in zip(keys, embeddings):
            self.cache[key] = emb
    
    def get_size(self) -> int:
        """Devuelve el tamaño máximo de la caché"""
        return self.cache.maxsize
    
    def get_usage(self) -> int:
        """Devuelve el número actual de elementos en caché"""
        return len(self.cache)