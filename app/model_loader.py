from sentence_transformers import SentenceTransformer
import os
import torch
import numpy as np
import logging
import psutil

logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self, torch_config):
        model_name = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2")
        
        # Configuración de dispositivo basada en disponibilidad
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Ajustar precisión según configuración
        self.precision = "fp16" if self.device == "cuda" else "fp32"
        if torch_config.get("quantize", False) and self.device == "cpu":
            self.precision = "quantized"
        
        logger.info(f"⏳ Loading model {model_name} on {self.device} ({self.precision})...")
        
        # Cargar con optimizaciones
        load_params = {
            "device": self.device,
            "cache_folder": "./model_cache"
        }
        
        # Aplicar optimizaciones específicas
        if self.precision == "fp16" and self.device == "cuda":
            load_params["torch_dtype"] = torch.float16
        
        self.model = SentenceTransformer(model_name, **load_params)
        
        # Cuantización para CPU si se requiere
        if self.precision == "quantized":
            try:
                logger.info("🔄 Applying model quantization...")
                from optimum.onnxruntime import ORTModelForFeatureExtraction
                self.model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
            except ImportError:
                logger.warning("⚠️ ONNX Runtime not installed, skipping quantization")
                self.precision = "fp32"
        
        logger.info(f"✅ Model loaded | Dims: {self.model.get_sentence_embedding_dimension()}")
    
    def encode(self, texts, **kwargs):
        """Método optimizado con manejo automático de recursos"""
        # Configuración automática de batch_size
        if "batch_size" not in kwargs:
            mem = psutil.virtual_memory()
            free_mem = mem.available
            text_mem = len(texts) * 768 * 4  # 4 bytes por float
            safe_batch = max(1, min(len(texts), int(free_mem / (text_mem * 2.5))))
            kwargs["batch_size"] = safe_batch
        
        # Manejo de precisión
        if self.precision == "fp16" and self.device == "cuda":
            kwargs["convert_to_tensor"] = True
            embeddings = self.model.encode(texts, **kwargs)
            return embeddings.half().cpu().numpy()
        
        return self.model.encode(texts, **kwargs)