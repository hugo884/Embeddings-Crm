from cachetools import TTLCache
import hashlib
from typing import List, Optional, Dict
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class EmbeddingCache:
    def __init__(self, max_size=10000, ttl=3600):
        self.cache = TTLCache(maxsize=max_size, ttl=ttl)
        self.stats = defaultdict(int)  # Para rastrear hits y misses
        logger.info(f"🆕 Caché inicializada | Tamaño: {max_size} | TTL: {ttl}s")
    
    def generate_key(self, text: str) -> str:
        """Genera clave única para el texto normalizado"""
        normalized = text.strip().lower()
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def get_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Obtiene embeddings para un lote de textos"""
        keys = [self.generate_key(text) for text in texts]
        results = []
        
        for key in keys:
            item = self.cache.get(key)
            if item is not None:
                self.stats['hits'] += 1
                results.append(item)
            else:
                self.stats['misses'] += 1
                results.append(None)
        
        logger.debug(f"🔍 Cache lookup | Texts: {len(texts)} | Hits: {self.stats['hits']} | Misses: {self.stats['misses']}")
        return results
    
    def set_batch(self, texts: List[str], embeddings: List[List[float]]):
        """Almacena embeddings para un lote de textos"""
        if len(texts) != len(embeddings):
            logger.error("❌ Mismatch entre textos y embeddings al almacenar en caché")
            return
            
        for text, emb in zip(texts, embeddings):
            key = self.generate_key(text)
            self.cache[key] = emb
        
        logger.debug(f"💾 Cache store | Items: {len(texts)}")
    
    def get_stats(self) -> Dict[str, int]:
        """Devuelve estadísticas de la caché"""
        return {
            "max_size": self.cache.maxsize,
            "current_size": len(self.cache),
            "hits": self.stats['hits'],
            "misses": self.stats['misses']
        }