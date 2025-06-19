from pydantic import BaseModel
from typing import List

class EmbedItemRequest(BaseModel):
    productId: int
    text: str

class EmbedItemResponse(BaseModel):
    productId: int
    embedding: List[float]

class EmbedRequest(BaseModel):
    items: List[EmbedItemRequest]

class EmbedResponse(BaseModel):
    items: List[EmbedItemResponse]
    cached: int
    processing_time: float
    batch_size: int