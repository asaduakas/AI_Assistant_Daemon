from app.indexing.index_project import embed_one_document
from llama_cpp.llama import Llama
from app.config import settings

#---EMBEDDING MODEL---
embedding_model = Llama(
    model_path=str(settings.MODEL_PATH),
    embedding=True,
    n_gpu_layers=settings.N_GPU_LAYERS,
    n_threads=settings.N_THREADS,
    n_ctx=settings.N_CTX,
    n_batch=settings.N_BATCH,
    verbose=False,
)

if __name__ == "__main__":
    emb = embed_one_document("hello world", embedding_model)
    print(len(emb))

