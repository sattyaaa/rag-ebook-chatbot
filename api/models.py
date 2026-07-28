from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: Optional[str] = None


class RetrievedChunk(BaseModel):
    text: str
    source: str
    page: Optional[int] = None
    score: float


class ChatResponse(BaseModel):
    final_answer: str
    confidence: float
    retrieved_context_chunks: List[RetrievedChunk]
