from sentence_transformers import SentenceTransformer
import os

class EmbeddingModel:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls._instance.load_model()
        return cls._instance
    
    def load_model(self):
        model_name = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2")
        print(f"⏳ Loading model {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"✅ Model loaded | Dims: {self.model.get_sentence_embedding_dimension()}")
    
    def embed(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)