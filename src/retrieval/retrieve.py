from typing import List
from src.vectorstore.base import VectorStore
from src.core.embeddings import embed_one_document
from src.core.model_loader import get_embedding_model
from src.config import settings

def retrieve_chunks(
        query: str,
        project_id: str,
        vector_store: VectorStore,
        k: int = 5
) -> List[str]:
    """
    Retrieve top-k relevant code chunks for a query.
    Vector-store-agnostic.
    """
    embedding_model = get_embedding_model()

    query_embedding = embed_one_document(
        query,
        embedding_model=embedding_model,
    )

    if not query_embedding:
        return []

    # Chroma requires the top-level `where` to contain a single operator
    # when combining multiple conditions. Use `$and` to combine filters.
    results = vector_store.query(
        query_embedding=query_embedding,
        k=k,
        where={
            "project_id": project_id,
            "chunk_kind": "method",
        }
    )

    documents = results.get("documents", [])

    return documents