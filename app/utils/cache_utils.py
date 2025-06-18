from cachetools import TTLCache
import hashlib

class EmbeddingCache:
    def __init__(self, max_size=10000, ttl=3600):
        self.cache = TTLCache(maxsize=max_size, ttl=ttl)
    
    def generate_key(self, text):
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def get(self, text):
        key = self.generate_key(text)
        return self.cache.get(key)
    
    def set(self, text, embedding):
        key = self.generate_key(text)
        self.cache[key] = embedding