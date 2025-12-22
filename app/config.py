from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    
    ROOT_DIR = Path.home() / "AI_Assistant_Daemon"
    PROJECT_ID = "AI_Assistant_Daemon"
    SUPPORTED_EXTENSIONS = {".py"}

    INCLUDE_DIRS ={
        "app",
        "daemon.py",
        "tuning",
    }

    #---EMBEDDING MODEL---
    MODEL_PATH = Path("models/mistral-7b-instruct-v0.1.Q5_K_M.gguf")
    EMBED_DIM = 4096   
    
    #llama.cpp runtime
    N_GPU_LAYERS = 32
    N_THREADS = 6
    N_CTX: int = 4096
    N_BATCH=256
    MAX_TOKENS = 3800

    #Chroma persistent storage
    CHROMA_DIR: Path = Path("chroma-persist")
    COLLECTION_CHUNKS: str = "chunks_v1"

settings = Settings()