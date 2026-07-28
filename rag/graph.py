from functools import lru_cache
import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import END, START, StateGraph

from rag.config import settings
from rag.models import SourceChunk, RagState
from rag.prompts import SYSTEM_PROMPT, HYDE_PROMPT, GREETING_CLASSIFY_PROMPT
from rag.model_factory import get_embeddings, get_llm, get_rewriter_llm, get_router_llm

# Configure logger
logger = logging.getLogger("rag.graph")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@lru_cache(maxsize=1)
def get_vectorstore() -> PineconeVectorStore:
    return PineconeVectorStore.from_existing_index(
        index_name=settings.pinecone_index_name,
        embedding=get_embeddings(),
        namespace=settings.pinecone_namespace,
    )


def retrieve(state: RagState) -> Dict[str, Any]:
    logger.info("Running Node: retrieve")
    query = state.get("search_query", state["question"])
    words = query.split()
    word_count = len(words)
    preview = " ".join(words[:10]) + "..." if word_count > 10 else query
    logger.info("Querying Pinecone vector index with query '%s' (%d words) for top_%s candidate chunks...", preview, word_count, settings.top_k)
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_relevance_scores(
        query,
        k=settings.top_k,
    )
    logger.info("Retrieved %s chunks from Pinecone.", len(results))

    chunks: List[SourceChunk] = []
    context_parts: List[str] = []
    best_score = 0.0

    for doc, score in results:
        meta = doc.metadata or {}
        source = str(meta.get("source", settings.pdf_path))
        page = meta.get("page")
        try:
            page_num = int(float(page)) if page is not None else None
        except (ValueError, TypeError):
            page_num = None

        chunk = {
            "text": doc.page_content.strip(),
            "source": source,
            "page": page_num,
            "score": round(float(score), 4),
        }
        chunks.append(chunk)
        best_score = max(best_score, float(score))

        header = f"[source: {source}"
        if page_num is not None:
            header += f" | page: {page_num}"
        header += f" | score: {float(score):.3f}]"
        context_parts.append(f"{header}\n{doc.page_content.strip()}")

    return {
        "chunks": chunks,
        "context": "\n\n---\n\n".join(context_parts),
        "confidence": round(best_score, 4),
    }


def generate_hyde(state: RagState) -> Dict[str, Any]:
    logger.info("Running Node: generate_hyde")
    question = state["question"]
    prompt = HYDE_PROMPT.format(question=question)

    try:
        response = get_rewriter_llm().invoke(prompt)
        search_query = (response.content or "").strip()
        if not search_query:
            search_query = question
    except Exception as e:
        logger.warning("HyDE document generation failed: %s. Falling back to original question.", e)
        search_query = question

    logger.info("Original question: '%s'", question)
    words = search_query.split()
    word_count = len(words)
    preview = " ".join(words[:10]) + "..." if word_count > 10 else search_query
    logger.info("Generated HyDE document: '%s' (%d words)", preview, word_count)
    return {"search_query": search_query}





def route(state: RagState) -> str:
    logger.info("Evaluating Conditional Edge: route")
    confidence = float(state.get("confidence", 0.0))
    min_conf = settings.min_confidence
    if not state.get("chunks"):
        logger.info("Routing decision: fallback (No chunks present)")
        return "fallback"
    if confidence < min_conf:
        logger.info("Routing decision: fallback (Confidence %.4f < threshold %.2f)", confidence, min_conf)
        return "fallback"
    logger.info("Routing decision: generate (Confidence %.4f >= threshold %.2f)", confidence, min_conf)
    return "generate"


def generate(state: RagState) -> Dict[str, Any]:
    logger.info("Running Node: generate")
    logger.info("Requesting response from generator model: %s...", settings.groq_chat_model)
    llm = get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Question:\n{state['question']}\n\n"
                f"Context:\n{state.get('context', '')}\n\n"
                "Answer:"
            )
        ),
    ]
    response = llm.invoke(messages)
    answer = (response.content or "").strip()

    if not answer:
        answer = "I don't know based on the provided PDF. What can I help you with? Please ask questions from the PDF."

    return {
        "answer": answer,
        "final_answer": answer,
    }


def fallback(_: RagState) -> Dict[str, Any]:
    logger.info("Running Node: fallback")
    logger.info("Context confidence was insufficient. Instantly returning fallback message.")
    answer = "I don't know based on the provided PDF. What can I help you with? Please ask questions from the PDF."
    return {
        "answer": answer,
        "final_answer": answer,
    }


def greeting_node(state: RagState) -> Dict[str, Any]:
    logger.info("Running Node: greeting_node")
    answer = "Hello! How can I help you today? Please ask me any questions about the PDF or book."
    return {
        "answer": answer,
        "final_answer": answer,
    }


def route_input(state: RagState) -> str:
    logger.info("Evaluating Entry Edge: route_input")
    question = (state.get("question") or "").strip()
    
    # LLM classification using the new router model
    prompt = GREETING_CLASSIFY_PROMPT.format(question=question)
    try:
        response = get_router_llm().invoke(prompt)
        decision = (response.content or "").strip().lower()
        logger.info("Router LLM raw response: %s", decision)
        if "greet" in decision:
            logger.info("LLM-path: Input classified as a greeting.")
            return "greet"
    except Exception as e:
        logger.warning("Router LLM call failed: %s. Defaulting to RAG query flow.", e)
        
    logger.info("Input classified as a query. Routing to HyDE.")
    return "rag"


def build_graph():
    builder = StateGraph(RagState)

    builder.add_node("greeting_node", greeting_node)
    builder.add_node("generate_hyde", generate_hyde)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("fallback", fallback)

    builder.add_conditional_edges(
        START,
        route_input,
        {
            "greet": "greeting_node",
            "rag": "generate_hyde",
        },
    )
    builder.add_edge("greeting_node", END)
    builder.add_edge("generate_hyde", "retrieve")
    builder.add_conditional_edges(
        "retrieve",
        route,
        {
            "generate": "generate",
            "fallback": "fallback",
        },
    )
    builder.add_edge("generate", END)
    builder.add_edge("fallback", END)

    return builder.compile()


rag_graph = build_graph()
