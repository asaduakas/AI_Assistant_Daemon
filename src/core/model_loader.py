from llama_cpp import Llama
from src.config import settings
import threading
from typing import Optional

_llm: Optional[Llama] = None
_lock = threading.Lock()


def get_embedding_model() -> Llama:
    """Return a singleton Llama instance configured for embeddings.

    This is thread-safe and lazy-initialized so importing modules don't
    immediately allocate GPU/CPU resources.
    """
    global _llm
    if _llm is None:
        with _lock:
            if _llm is None:
                _llm = Llama(
                    model_path=str(settings.MODEL_PATH),
                    embedding=True,
                    n_gpu_layers=settings.N_GPU_LAYERS,
                    n_threads=settings.N_THREADS,
                    n_ctx=settings.N_CTX,
                    n_batch=settings.N_BATCH,
                    verbose=False,
                )
    return _llm
