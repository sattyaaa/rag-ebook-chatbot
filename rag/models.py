from typing import List, Optional, TypedDict

class SourceChunk(TypedDict):
    text: str
    source: str
    page: Optional[int]
    score: float


class RagState(TypedDict, total=False):
    question: str
    search_query: str
    context: str
    chunks: List[SourceChunk]
    confidence: float
    answer: str
    final_answer: str
