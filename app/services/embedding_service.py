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

    async def embed_texts(self, texts: List[str]) -> Tuple[List[List[float]], int, float, int]:
        """
        Devuelve:
            - embeddings: Listado de vectores (floats)
            - cached_count: cuántos entraron por caché
            - total_time: tiempo total de la llamada
            - used_batch: número de textos procesados en batch (no-caché)
        """
        logger.info(f"🔍 Iniciando embed_texts para {len(texts)} textos")
        start_time = time.time()

        embeddings: List[List[float]] = []
        uncached: List[str] = []
        cache_map: dict[str, int] = {}

        try:
            # 1) Leer caché
            for idx, txt in enumerate(texts):
                cached = self.cache.get(txt)
                if cached is not None:
                    embeddings.append(cached.tolist())
                else:
                    embeddings.append(None)          # placeholder
                    cache_map[txt] = idx
                    uncached.append(txt)

            hits = len(texts) - len(uncached)
            logger.info(f"✅ Caché: {hits} hits, {len(uncached)} misses")

            used_batch = 0
            # 2) Procesar no-cache en batches si existen
            if uncached:
                free_mem = psutil.virtual_memory().available
                # ejemplo de batch size dinámico
                batch_size = min(len(uncached), max(1, int(free_mem / (768 * 4 * 1500))))
                batches = [
                    uncached[i : i + batch_size]
                    for i in range(0, len(uncached), batch_size)
                ]
                logger.info(f"🚀 Procesando {len(uncached)} textos en {len(batches)} batches (batch_size={batch_size})")

                loop = __import__('asyncio').get_running_loop()
                tasks = [
                    loop.run_in_executor(
                        self.executor,
                        lambda bn=batch: self.model.encode(
                            bn,
                            batch_size=len(bn),
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                            show_progress_bar=False
                        )
                    )
                    for batch in batches
                ]
                results = await __import__('asyncio').gather(*tasks)
                all_emb = np.vstack(results)

                # 3) Guardar en caché y completar embeddings
                for i, txt in enumerate(uncached):
                    pos = cache_map[txt]
                    emb = all_emb[i]
                    embeddings[pos] = emb.tolist()
                    self.cache.set(txt, emb)

                used_batch = len(uncached)
                logger.info(f"✅ Batch completado, almacenados {used_batch} embeddings en caché")

            # Cálculo de tiempos
            total_time = time.time() - start_time
            logger.info(f"⏱️ Tiempo total embed_texts: {total_time:.2f}s")
            return embeddings, hits, total_time, used_batch

        except Exception as e:
            logger.error(f"❌ Error en embed_texts: {e}", exc_info=True)
            # Propagar la excepción para que el controller convierta en HTTPException si aplica
            raise
