from typing import List

from fastapi import FastAPI

from rag.graph import rag_graph
from rag.ingest import ingest_pdf
from api.models import ChatRequest, ChatResponse, RetrievedChunk


app = FastAPI(
    title="Agentic AI eBook RAG Chatbot",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest():
    total_chunks = ingest_pdf(reset_namespace=True)
    return {
        "status": "success",
        "message": f"Ingested {total_chunks} chunks into Pinecone.",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    result = rag_graph.invoke({"question": payload.question})

    chunks = result.get("chunks", [])
    return {
        "final_answer": result.get("final_answer", "I don't know based on the provided PDF. What can I help you with? Please ask questions from the PDF."),
        "confidence": float(result.get("confidence", 0.0)),
        "retrieved_context_chunks": chunks,
    }
