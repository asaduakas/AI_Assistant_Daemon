from pathlib import Path
from llama_cpp.llama import Llama
from app.config import settings

from app.indexing.index_project import (
    read_file,
    python_method_chunks,
    embed_one_document,
    normalize_embeddings,
)

# --- EMBEDDING MODEL ---
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
    path = Path("/home/ace/AI_Assistant_Daemon/app/indexing/index_project.py")

    source = read_file(path)

    chunks = list(python_method_chunks(source))

    print(f"Found {len(chunks)} chunks\n")

    # Pick ONE chunk
    chunk_text, start, end, symbol = chunks[0]

    print("=== CHUNK METADATA ===")
    print("Symbol:", symbol)
    print("Lines:", start, "-", end)
    print("Chars:", len(chunk_text))
    print()

    print("=== CHUNK TEXT ===")
    print(chunk_text[:500], "...\n")

    # --- RAW EMBEDDING ---
    raw = embedding_model.embed(chunk_text)

    print("=== RAW EMBEDDING TYPE ===")
    print(type(raw))
    if isinstance(raw, tuple):
        print("Tuple contents types:", [type(x) for x in raw])
    print()

    # --- NORMALIZATION ---
    normalized = normalize_embeddings(raw)

    print("=== NORMALIZED ===")
    print("Docs:", len(normalized))
    print("Vector dim:", len(normalized[0]))
    print("First 10 values:", normalized[0][:10])
