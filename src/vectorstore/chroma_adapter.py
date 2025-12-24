from typing import Any, Dict, List
from src.vectorstore.base import VectorStore
from src.vectorstore.chroma_store import get_chunks_collection

class ChromaVectorStore(VectorStore):
    def __init__(self):
        self.collection = get_chunks_collection()

    def upsert(self, ids, embeddings, documents, metadatas) -> None:
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def query(self, query_embedding, k, where=None):
        # Translate generic filters → Chroma filters
        chroma_where = None
        if where:
            chroma_where = {"$and": [{k: v} for k, v in where.items()]}

        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=chroma_where,
        )

        # Normalize Chroma's weird return shape
        return {
            "documents": (raw.get("documents") or [[]])[0],
            "ids": (raw.get("ids") or [[]])[0],
            "metadatas": (raw.get("metadatas") or [[]])[0],
            "scores": (raw.get("distances") or [[]])[0],
        }

    def delete(self, where):
        chroma_where = {"$and": [{k: v} for k, v in where.items()]}
        self.collection.delete(where=chroma_where)