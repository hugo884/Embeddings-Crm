from sentence_transformers import SentenceTransformer
import os
import torch
import logging
import psutil
import numpy as np
from typing import List

logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self, torch_config=None):
        model_name = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2")
        
        # Determinar dispositivo automáticamente
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Manejo de precisión más robusto
        self.precision = self._determine_precision(torch_config)
        
        logger.info(f"⏳ Loading model '{model_name}' on {self.device} ({self.precision})...")
        
        # Parámetros de carga optimizados
        load_params = {
            "device": self.device,
            "cache_folder": "./model_cache"
        }
        
        # Configuración de precisión para GPU
        if self.precision == "fp16" and self.device == "cuda":
            load_params["torch_dtype"] = torch.float16
        
        # Cargar modelo principal
        self.model = SentenceTransformer(model_name, **load_params)
        self.model_dimensions = self.model.get_sentence_embedding_dimension()
        
        # Manejo de cuantización más seguro
        self.quantized_model = None
        if self.precision == "quantized":
            self._initialize_quantized_model(model_name)
        
        logger.info(f"✅ Model loaded | Dimensions: {self.model_dimensions}")

    def _determine_precision(self, torch_config):
        """Determina la precisión óptima basada en configuración y hardware"""
        if self.device == "cuda":
            return "fp16"
        
        if torch_config and torch_config.get("quantize", False):
            return "quantized"
        
        return "fp32"

    def _initialize_quantized_model(self, model_name):
        """Inicializa modelo cuantizado solo si es necesario y posible"""
        try:
            logger.info("🔄 Applying model quantization...")
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            self.quantized_model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
            logger.info("✅ Quantization applied successfully")
        except ImportError:
            logger.warning("⚠️ ONNX Runtime not installed, using standard model")
            self.precision = "fp32"
        except Exception as e:
            logger.error(f"⚠️ Quantization failed: {str(e)}")
            self.precision = "fp32"

    def _calculate_optimal_batch_size(self, texts: List[str]) -> int:
        """Calcula el tamaño de lote óptimo basado en memoria disponible"""
        mem = psutil.virtual_memory()
        free_mem = mem.available
        
        # Estimación de memoria requerida (4 bytes por float * dimensiones * longitud promedio)
        avg_text_len = sum(len(t) for t in texts) / max(1, len(texts))
        estimated_mem_per_text = self.model_dimensions * 4 * (avg_text_len / 4)  # Asume 4 caracteres ≈ 1 token
        
        # Cálculo seguro considerando memoria libre
        max_batch_size = max(1, int(free_mem / (estimated_mem_per_text * 2.5)))  # Factor de seguridad 2.5x
        
        # Limitar por cantidad de textos y máximo razonable
        return min(len(texts), max(8, min(128, max_batch_size)))

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Codifica textos con manejo automático de recursos"""
        # Calcular batch_size óptimo si no se especifica
        if "batch_size" not in kwargs:
            kwargs["batch_size"] = self._calculate_optimal_batch_size(texts)
            logger.debug(f"Using automatic batch size: {kwargs['batch_size']}")
        
        # Usar modelo cuantizado si está disponible
        if self.precision == "quantized" and self.quantized_model:
            return self.quantized_model.encode(texts, **kwargs)
        
        # Manejo de precisión para GPU
        if self.precision == "fp16" and self.device == "cuda":
            kwargs["convert_to_tensor"] = True
            embeddings = self.model.encode(texts, **kwargs)
            return embeddings.half().cpu().numpy()
        
        # Caso estándar
        return self.model.encode(texts, **kwargs)
    
    def get_dimensions(self) -> int:
        """Devuelve las dimensiones del embedding"""
        return self.model_dimensions