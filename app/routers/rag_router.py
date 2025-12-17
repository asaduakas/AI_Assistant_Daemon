from fastapi import APIRouter
from app.retrieval.chroma_store import collection_stats, get_chunks_collection

router = APIRouter(prefix="/rag", tags=["rag"])

@router.get("/health")
def rag_health():
    _= get_chunks_collection()
    return {"ok": True, "stats": collection_stats()}