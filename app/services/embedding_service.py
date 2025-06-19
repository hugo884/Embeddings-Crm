import asyncio
import time
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
        self.dims = model.get_dimensions()
        logger.info("🛠️ EmbeddingService inicializado con modelo y caché")

    async def embed_texts(self, texts: List[str]) -> Tuple[List[List[float]], int, float, int]:
        """
        Procesa textos devolviendo embeddings en el mismo orden de entrada
        Devuelve:
            - embeddings: Listado de vectores en el orden original
            - cached_count: textos servidos desde caché
            - total_time: tiempo total en segundos
            - processed_count: textos procesados (no desde caché)
        """
        logger.debug(f"📥 Recibida solicitud para {len(texts)} textos")
        start_time = time.perf_counter()
        total_texts = len(texts)
        
        try:
            # 1) Buscar en caché
            cache_results = self.cache.get_batch(texts)
            
            # 2) Preparar estructuras para resultados
            embeddings = []         # Resultados finales
            to_process = []         # Textos a procesar
            process_positions = []  # Posiciones originales de los textos a procesar
            cached_count = 0

            for idx, (text, cached_emb) in enumerate(zip(texts, cache_results)):
                if cached_emb is not None:
                    embeddings.append(cached_emb)
                    cached_count += 1
                else:
                    # Placeholder temporal (será reemplazado después)
                    embeddings.append(None)
                    to_process.append(text)
                    process_positions.append(idx)
            
            logger.info(f"✅ Caché: {cached_count} hits | Misses: {len(to_process)}")
            processed_count = 0
            
            # 3) Procesar textos no encontrados en caché
            if to_process:
                processed_embeddings = await self.process_uncached_texts(to_process)
                processed_count = len(processed_embeddings)
                
                # Actualizar resultados con embeddings generados
                for pos, emb in zip(process_positions, processed_embeddings):
                    embeddings[pos] = emb
                
                # Almacenar nuevos embeddings en caché
                self.cache.set_batch(to_process, processed_embeddings)
            
            # 4) Validar que no quedan placeholders
            for i, emb in enumerate(embeddings):
                if emb is None:
                    logger.warning(f"⚠️ Embedding nulo en posición {i}, usando vector cero")
                    embeddings[i] = [0.0] * self.dims
            
            # 5) Métricas finales
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
            # Fallback seguro: embeddings de ceros
            return [[0.0] * self.dims] * len(texts), 0, 0, 0

    async def process_uncached_texts(self, texts: List[str]) -> List[List[float]]:
        """Procesa textos no encontrados en caché y devuelve embeddings"""
        logger.info(f"⚙️ Procesando {len(texts)} textos nuevos")
        start_time = time.perf_counter()
        
        try:
            # 1) Calcular tamaño óptimo de batch
            batch_size = self.calculate_optimal_batch_size(texts)
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
            all_embeddings = []
            
            logger.debug(f"🔢 Batch config: {len(batches)} batches de {batch_size} textos")
            
            # 2) Procesar cada batch
            loop = asyncio.get_running_loop()
            
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
                    
                    # Convertir y almacenar resultados
                    processed = self.convert_embeddings(batch_embeddings)
                    all_embeddings.extend(processed)
                    
                    batch_time = time.perf_counter() - batch_start
                    logger.debug(f"✅ Batch #{batch_idx+1} completado en {batch_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"⚠️ Error en batch #{batch_idx+1}: {str(e)}", exc_info=True)
                    # Fallback: embeddings de ceros para este batch
                    all_embeddings.extend([[0.0] * self.dims] * len(batch))
            
            # 3) Validar longitud de resultados
            if len(all_embeddings) != len(texts):
                logger.error(
                    f"❌ Mismatch en embeddings generados: "
                    f"Esperados {len(texts)}, Obtenidos {len(all_embeddings)}"
                )
                # Ajustar longitud si es necesario
                if len(all_embeddings) < len(texts):
                    all_embeddings.extend([[0.0] * self.dims] * (len(texts) - len(all_embeddings)))
                else:
                    all_embeddings = all_embeddings[:len(texts)]
            
            # 4) Métricas de procesamiento
            process_time = time.perf_counter() - start_time
            logger.info(
                f"✨ {len(texts)} textos procesados en {process_time:.2f}s | "
                f"Tasa: {len(texts)/process_time:.1f} textos/s"
            )
            
            return all_embeddings
        
        except Exception as e:
            logger.error(f"🔥 Error procesando textos: {str(e)}", exc_info=True)
            return [[0.0] * self.dims] * len(texts)

    def calculate_optimal_batch_size(self, texts: List[str]) -> int:
        """Calcula el tamaño óptimo de batch basado en recursos disponibles"""
        try:
            # Obtener métricas del sistema
            free_mem = psutil.virtual_memory().available
            cpu_count = psutil.cpu_count(logical=False) or 1
            
            # Calcular memoria requerida
            total_chars = sum(len(t) for t in texts)
            avg_text_len = total_chars / max(1, len(texts))
            
            # Estimación de memoria (conservadora)
            est_mem_per_char = self.dims * 0.5  # bytes por carácter
            est_mem_per_text = avg_text_len * est_mem_per_char
            est_total_mem = est_mem_per_text * len(texts)
            
            # Calcular batch size basado en memoria
            mem_based = max(1, int(free_mem / (est_mem_per_text * 2.5)))  # Factor de seguridad
            
            # Considerar capacidad de CPU
            cpu_based = cpu_count * 8
            
            # Limitar por parámetros prácticos
            batch_size = min(
                len(texts), 
                max(8, min(256, min(mem_based, cpu_based)))
            )
            
            logger.debug(
                f"🧮 Batch size calculado: "
                f"Mem libre: {free_mem/1024**2:.1f}MB, "
                f"Mem estimada: {est_total_mem/1024**2:.1f}MB, "
                f"CPU: {cpu_count}, "
                f"Resultado: {batch_size}"
            )
            
            return batch_size
        
        except Exception as e:
            logger.warning(f"⚠️ Error calculando batch size: {e}. Usando valor por defecto")
            return min(32, len(texts))

    def convert_embeddings(self, embeddings) -> List[List[float]]:
        """Convierte embeddings a lista de listas de floats de forma robusta"""
        try:
            # Caso 1: numpy array
            if isinstance(embeddings, np.ndarray):
                return embeddings.tolist()
            
            # Caso 2: lista de arrays
            if isinstance(embeddings, list) and all(isinstance(e, np.ndarray) for e in embeddings):
                return [e.tolist() for e in embeddings]
            
            # Caso 3: tensor u otro tipo
            if hasattr(embeddings, "tolist"):
                return embeddings.tolist()
            
            # Caso 4: ya es lista de listas
            if isinstance(embeddings, list) and all(isinstance(e, list) for e in embeddings):
                return embeddings
            
            # Formato desconocido
            logger.warning("⚠️ Formato de embeddings desconocido, forzando conversión")
            return [list(map(float, emb)) for emb in embeddings]
        
        except Exception as e:
            logger.error(f"❌ Error convirtiendo embeddings: {e}", exc_info=True)
            return [[0.0] * self.dims] * (len(embeddings) if hasattr(embeddings, "__len__") else 1)