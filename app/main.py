from fastapi import FastAPI
import uvicorn

from app.router_llm import router as llm_router

app = FastAPI(title="Local AI Assistant Daemon")
app.include_router(llm_router)

def start_server():
    uvicorn.run(app, host="127.0.0.1", port=8000)