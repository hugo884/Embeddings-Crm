import asyncio
import time
import math
import numpy as np
import psutil
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from ..utils.cache_utils import EmbeddingCache
from ..model_loader import EmbeddingModel

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(
        self,
        model: EmbeddingModel,
        cache: EmbeddingCache,
        executor: ThreadPoolExecutor
    ):
        self.model = model
        self.cache = cache
        self.executor = executor
        logger.info("🛠️ EmbeddingService inicializado con modelo y caché")

    async def embed_texts(self, texts: List[str]) -> Tuple[List[List[float]], int, float, int]:
        """
        Procesa textos devolviendo embeddings, contando caché hits y midiendo tiempos
        Devuelve:
            - embeddings: Listado de vectores
            - cached_count: textos servidos desde caché
            - total_time: tiempo total en segundos
            - processed_count: textos procesados (no desde caché)
        """
        logger.debug(f"📥 Recibida solicitud para {len(texts)} textos")
        start_time = time.perf_counter()
        total_texts = len(texts)
        
        try:
            # 1) Leer caché en lote
            logger.debug("🔍 Buscando en caché...")
            cache_results = self.cache.get_batch(texts)
            
            embeddings = []
            uncached_texts = []
            text_position_map = {}
            cached_count = 0

            for idx, (text, cached_emb) in enumerate(zip(texts, cache_results)):
                if cached_emb is not None:
                    embeddings.append(cached_emb)
                    cached_count += 1
                else:
                    embeddings.append(None)  # placeholder
                    text_position_map[text] = idx
                    uncached_texts.append(text)
            
            logger.info(f"✅ Caché: {cached_count} hits, {len(uncached_texts)} misses")
            processed_count = 0
            
            # 2) Procesar textos no encontrados en caché
            if uncached_texts:
                processed_count = await self.process_uncached_texts(
                    uncached_texts,
                    embeddings,
                    text_position_map
                )
            
            # Métricas finales
            total_time = time.perf_counter() - start_time
            logger.info(
                f"🏁 Procesados {total_texts} textos | "
                f"Caché: {cached_count} | "
                f"Nuevos: {processed_count} | "
                f"Tiempo: {total_time:.3f}s"
            )
            
            return embeddings, cached_count, total_time, processed_count
        
        except Exception as e:
            logger.error(f"❌ Error crítico en embed_texts: {str(e)}", exc_info=True)
            raise

    async def process_uncached_texts(
        self, 
        texts: List[str], 
        embeddings: List, 
        position_map: dict
    ) -> int:
        """Procesa textos no encontrados en caché y actualiza resultados"""
        logger.info(f"⚙️ Procesando {len(texts)} textos nuevos")
        start_time = time.perf_counter()
        
        try:
            # Calcular batch size óptimo
            batch_size = self.calculate_optimal_batch_size(texts)
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
            
            logger.debug(
                f"🔢 Batch config: textos={len(texts)}, "
                f"batch_size={batch_size}, "
                f"batches={len(batches)}"
            )
            
            # Procesar cada batch
            loop = asyncio.get_running_loop()
            all_embeddings = []
            
            for batch_idx, batch in enumerate(batches):
                logger.debug(f"🔄 Procesando batch #{batch_idx+1}/{len(batches)} ({len(batch)} textos)")
                batch_start = time.perf_counter()
                
                try:
                    # Ejecutar en thread pool
                    batch_embeddings = await loop.run_in_executor(
                        self.executor,
                        lambda b=batch: self.model.encode(
                            b,
                            batch_size=len(b),
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                            show_progress_bar=False
                        )
                    )
                    
                    # Convertir a formato serializable
                    processed_embeddings = self.convert_embeddings(batch_embeddings)
                    all_embeddings.extend(processed_embeddings)
                    
                    batch_time = time.perf_counter() - batch_start
                    logger.debug(
                        f"✅ Batch #{batch_idx+1} completado en {batch_time:.2f}s | "
                        f"{len(processed_embeddings)} embeddings generados"
                    )
                    
                except Exception as e:
                    logger.error(f"⚠️ Error en batch #{batch_idx+1}: {str(e)}", exc_info=True)
                    # Insertar placeholders para mantener consistencia
                    all_embeddings.extend([[0.0]*self.model.get_dimensions()] * len(batch))
            
            # Actualizar caché y resultados
            logger.debug("💾 Almacenando embeddings en caché...")
            self.cache.set_batch(texts, all_embeddings)
            
            for text, emb in zip(texts, all_embeddings):
                position = position_map[text]
                embeddings[position] = emb
            
            process_time = time.perf_counter() - start_time
            logger.info(
                f"✨ {len(texts)} textos procesados en {process_time:.2f}s | "
                f"Tasa: {len(texts)/process_time:.1f} textos/s"
            )
            
            return len(texts)
        
        except Exception as e:
            logger.error(f"🔥 Error procesando textos no cacheados: {str(e)}", exc_info=True)
            return 0

    def calculate_optimal_batch_size(self, texts: List[str]) -> int:
        """Calcula el tamaño óptimo de batch basado en recursos disponibles"""
        try:
            # Obtener métricas del sistema
            free_mem = psutil.virtual_memory().available
            dims = self.model.get_dimensions()
            
            # Calcular memoria requerida
            total_chars = sum(len(t) for t in texts)
            avg_text_len = total_chars / max(1, len(texts))
            
            # Estimación conservadora: 4 caracteres = 1 token
            # Cada token: dims * 4 bytes (float32)
            est_mem_per_text = (avg_text_len / 4) * dims * 4
            
            # Calcular batch size seguro
            safe_batch_size = max(1, int(free_mem / (est_mem_per_text * 3.0)))  # Factor de seguridad 3x
            
            # Limitar por máximo de textos y límite práctico
            batch_size = min(
                len(texts), 
                max(8, min(256, safe_batch_size))  # Entre 8 y 256
            )
            
            logger.debug(
                f"🧮 Cálculo batch: "
                f"Mem libre={free_mem/1024**2:.1f}MB, "
                f"Mem/texto={est_mem_per_text:.1f} bytes, "
                f"Batch size={batch_size}"
            )
            
            return batch_size
        
        except Exception as e:
            logger.warning(f"⚠️ Error calculando batch size: {e}. Usando valor por defecto")
            return min(32, len(texts))

    def convert_embeddings(self, embeddings) -> List[List[float]]:
        """Convierte embeddings a lista de listas de floats"""
        try:
            if isinstance(embeddings, np.ndarray):
                return embeddings.tolist()
            elif hasattr(embeddings, "tolist"):
                return [emb.tolist() for emb in embeddings]
            elif isinstance(embeddings, list):
                return embeddings
            else:
                logger.warning("⚠️ Formato de embeddings desconocido, forzando conversión")
                return [list(map(float, emb)) for emb in embeddings]
        except Exception as e:
            logger.error(f"❌ Error convirtiendo embeddings: {e}", exc_info=True)
            dims = self.model.get_dimensions()
            return [[0.0] * dims] * len(embeddings)