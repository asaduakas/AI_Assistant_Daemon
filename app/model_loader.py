from llama_cpp import Llama
from app.config import settings

_llm: Llama | None = None

def get_llm() -> Llama:
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=str(settings.MODEL_PATH),
            n_gpu_layers=settings.N_GPU_LAYERS,
            n_threads=settings.N_THREADS,
            n_ctx=settings.N_CTX,
            embedding=True,
            verbose=False
        )
    return _llm
