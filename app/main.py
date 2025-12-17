from fastapi import FastAPI
import uvicorn

#from app.routers.llm_router import router as llm_router
from app.routers.rag_router import router as rag_router

app = FastAPI(title="Local AI Assistant Daemon")
app.include_router(rag_router)

def start_server():
    uvicorn.run(app, host="127.0.0.1", port=8000)