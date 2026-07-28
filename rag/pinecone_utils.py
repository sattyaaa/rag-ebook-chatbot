from pinecone import Pinecone
from rag.config import settings


def get_pc() -> Pinecone:
    return Pinecone(api_key=settings.pinecone_api_key)


def clear_namespace() -> None:
    pc = get_pc()
    index = pc.Index(settings.pinecone_index_name)
    try:
        index.delete(delete_all=True, namespace=settings.pinecone_namespace)
    except Exception as e:
        if "Namespace not found" in str(e) or "404" in str(e):
            pass
        else:
            raise e

