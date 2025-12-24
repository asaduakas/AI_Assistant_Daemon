from __future__ import annotations

from pathlib import Path
import json
import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api import ClientAPI

from src.config import settings

import logging 

_client = None
_chunks: Collection | None = None

logger = logging.getLogger(__name__)

def _get_client() -> ClientAPI:
    global _client
    if _client is None:
        settings.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))
    return _client

def get_chunks_collection() -> Collection:
    """
    Single collection that supports:
     - global scope: no filter
     - project scope
     - bite scope
    """
    global _chunks
    if _chunks is None:
        client = _get_client()

        schema_meta = {
            "schema_version": "1",
            "collection_role": "project_rag_chunks",
            "id_format": "sha1(root|relpath|chunk_kind|symbol|start_line|end_line|mtime_ns)",
            "document": "chunk text",
            "embedding": "llama_cpp embedding vector",
            "metadata_keys": [
                "project_id",
                "root",
                "path",
                "relpath",
                "language",
                "chunk_kind",
                "symbol",
                "start_line",
                "end_line",
                "mtime_ns",
            ],
            "notes": "Scope is a query-time filter; no duplicated embeddings.",
        }

        # Chroma's Rust bindings only accept simple metadata value types
        # (bool, int, float, str, sparse vector). Convert complex values
        # (lists/dicts) to JSON strings so the binding can accept them.
        safe_meta: dict = {}
        for k, v in schema_meta.items():
            if isinstance(v, (str, bool, int, float)):
                safe_meta[k] = v
            else:
                safe_meta[k] = json.dumps(v)

        _chunks = client.get_or_create_collection(
            name=settings.COLLECTION_CHUNKS,
            metadata=safe_meta
        )

    return _chunks

def collection_stats() -> dict:
    col = get_chunks_collection()
    try:
        n = col.count()
    except Exception:
        n = None
    return {
        "path": str(settings.CHROMA_DIR),
        "collection": settings.COLLECTION_CHUNKS,
        "count": n,
        "schema_version": col.metadata.get("schema_version") if col.metadata else None,
    }