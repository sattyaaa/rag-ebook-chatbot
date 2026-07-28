from functools import lru_cache
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from rag.config import settings

@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Returns the local HuggingFace embedding model."""
    return HuggingFaceEmbeddings(
        model_name=settings.hf_embedding_model,
    )

@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """Returns the primary LLM generator model."""
    return ChatGroq(
        model=settings.groq_chat_model,
        temperature=0,
        api_key=settings.groq_api_key,
    )



@lru_cache(maxsize=1)
def get_rewriter_llm() -> ChatGroq:
    """Returns the lightweight LLM query rewriter/HyDE model."""
    return ChatGroq(
        model=settings.groq_hyde_model,
        temperature=0,
        api_key=settings.groq_api_key,
    )

@lru_cache(maxsize=1)
def get_router_llm() -> ChatGroq:
    """Returns the lightweight LLM router model."""
    return ChatGroq(
        model=settings.groq_router_model,
        temperature=0,
        api_key=settings.groq_api_key,
    )
