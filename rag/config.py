from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")

    groq_chat_model: str = "llama-3.3-70b-versatile"
    groq_hyde_model: str = "llama-3.1-8b-instant"
    groq_router_model: str = "llama-3.1-8b-instant"
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    pinecone_index_name: str = "agentic-ai-ebook"
    pinecone_namespace: str = "default"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    pdf_path: str = "data/ebook-agentic-ai.pdf"

    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 7
    min_confidence: float = 0.5


settings = Settings()

if not settings.groq_api_key:
    raise ValueError("GROQ_API_KEY is missing")

if not settings.pinecone_api_key:
    raise ValueError("PINECONE_API_KEY is missing")
