from pydantic import BaseModel
from typing import List

class EmbedRequest(BaseModel):
    texts: List[str]

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    cached: int
    processing_time: float
    batch_size: int
