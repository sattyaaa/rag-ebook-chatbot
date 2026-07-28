from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_experimental.text_splitter import SemanticChunker

from rag.config import settings
from rag.pinecone_utils import clear_namespace
from rag.model_factory import get_embeddings


def ingest_pdf(reset_namespace: bool = True) -> int:
    pdf_path = Path(settings.pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if reset_namespace:
        clear_namespace()

    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    splitter = SemanticChunker(
        get_embeddings(),
        breakpoint_threshold_type="percentile"
    )
    chunks = splitter.split_documents(pages)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["pdf_name"] = pdf_path.name

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=settings.pinecone_index_name,
        embedding=get_embeddings(),
        namespace=settings.pinecone_namespace,
    )

    vectorstore.add_documents(chunks)
    return len(chunks)


if __name__ == "__main__":
    total = ingest_pdf(reset_namespace=True)
    print(f"Ingested {total} chunks.")
