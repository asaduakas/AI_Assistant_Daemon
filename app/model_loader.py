from llama_cpp import Llama
from app.config import settings

llm = Llama(
    model_path=str(settings.MODEL_PATH),
    n_gpu_layers=settings.N_GPU_LAYERS,
    n_threads=settings.N_THREADS,
    logits_all=False,
    verbose=False,
)

def get_llm():
    return llm
