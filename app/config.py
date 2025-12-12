from pathlib import Path

class Settings:
    MODEL_PATH = Path("models/mistral-7b-instruct-v0.1.Q5_K_M.gguf")
    N_GPU_LAYERS = 32
    N_THREADS = 6

settings = Settings()