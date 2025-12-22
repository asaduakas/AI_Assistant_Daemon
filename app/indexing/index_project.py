import ast, hashlib, os
from pathlib import Path
from typing import Iterator, Tuple, List, Sequence, Any

from app.config import settings
from app.retrieval.chroma_store import get_chunks_collection
from llama_cpp import Llama

#---CONFIG---
ROOT_DIR = Path.home() / "AI_Assistant_Daemon"
PROJECT_ID = "AI_Assistant_Daemon"
SUPPORTED_EXTENSIONS = {".py"}

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

#---helpers---
def iter_source_files(root: Path) -> Iterator[Path]:
    """
    Returns an iterator from root to below.
    
    :param root: Spawning point
    :type root: Path
    :return: Iterator 
    :rtype: Iterator[Path]
    """
    for path in root.rglob("*"):
        if path.suffix in SUPPORTED_EXTENSIONS and path.is_file():
            yield path

def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def sha1(text:str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

#---CHUNKING (Python AST)---
def python_method_chunks(source: str) -> Iterator[Tuple[str, int, int, str]]:
    """
    Yields method/function-level chunks.

    For top-level functions:
        symbol = function_name

    For class methods:
        symbol = ClassName.method_name

    Does NOT emit whole-class chunks.
    """
    tree = ast.parse(source)
    lines = source.splitlines()

    #Top level functions
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", None)
            if end is None:
                continue

            chunk_text = "\n".join(lines[start - 1 : end])
            yield chunk_text, start, end, node.name

    #Class methods
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_name = node.name

            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = child.lineno
                    end = getattr(child, "end_lineno", None)
                    if end is None:
                        continue

                    chunk_text = "\n".join(lines[start - 1 : end])
                    symbol = f"{class_name}.{child.name}"
                    yield chunk_text, start, end, symbol

#--- INDEXING ---
def collect_chunks(root_dir: Path, project_id: str) -> Tuple[List[str], List[dict], List[str]]:
    #Outputs
    documents: List[str] = []
    metadatas: List[dict] = []
    ids: List[str] = []

    #Walking the filesystem with an iterator
    for path in iter_source_files(root_dir):
        if path.suffix != ".py":
            continue

        relpath = str(path.relative_to(root_dir))   #Safety precautions
        source = read_file(path)

        stat = path.stat()                          #System info about the file (mtime)

        chunks = python_method_chunks(source)

        for chunk_text, start, end, symbol in chunks:
            if not chunk_text.strip():
                continue

            chunk_kind = "method"
            chunk_id = sha1(
                f"{root_dir}|{relpath}|{chunk_kind}|{symbol}|{start}|{end}|{stat.st_mtime_ns}"
            )

            documents.append(chunk_text)
            ids.append(chunk_id)
            metadatas.append(
                {
                    "project_id": project_id,
                    "root": str(root_dir),
                    "path": str(path),
                    "relpath": relpath,
                    "language": "python",
                    "chunk_kind": chunk_kind,
                    "symbol": symbol,
                    "start_line": start,
                    "end_line": end,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )

    return documents, metadatas, ids

def normalize_embeddings(res: Any) -> List[List[float]]:
    """
    Normalize embedding model output into List[List[float]].

    Handles:
    - tuple returns
    - single-vector returns
    - batch returns
    - numpy arrays
    """
    # Unwrap (embeddings, metadata) tuples
    emb = res[0] if isinstance(res, tuple) else res

    # Convert numpy arrays to Python lists
    try:
        import numpy as np
        if isinstance(emb, np.ndarray):
            emb = emb.tolist()
    except Exception:
        pass

    if not emb:
        raise ValueError("Empty embedding result")

    # Single vector: [float, float, ...]
    if isinstance(emb[0], (int, float)):
        return [emb]

    # Batch of vectors: [[float, ...], [float, ...]]
    return list(emb)

def embed_one_document(doc: str, embedding_model: Any,) -> List[float]:
    """
    Embed a single document.

    Raises if embedding fails.
    """
    res = embedding_model.embed(doc)
    batch = normalize_embeddings(res)

    return batch[0]

def embed_batch(batch: List[str], embedding_model: Any,) -> List[List[float]]:
    """
    Try embedding a batch.
    If it fails, fall back to per-document embedding.
    """
    try:
        res = embedding_model.embed(batch)
        embeddings = normalize_embeddings(res)

        if len(embeddings) != len(batch):
            raise ValueError("Batch size mismatch")

        return embeddings

    except Exception as e:
        print(f"Batch failed ({len(batch)} docs): {e}")
        results: List[List[float]] = []

        for idx, doc in enumerate(batch):
            try:
                emb = embed_one_document(doc, embedding_model)
                results.append(emb)
            except Exception as e2:
                print(f"  Failed doc {idx} in batch: {e2}")
                results.append([])  # placeholder

        return results


def embed_documents(documents: List[str], embedding_model: Any, batch_size: int = 8,
                    ) -> Tuple[List[List[float]], List[int]]:
    """
    Embed documents in batches with safe fallback.

    Returns:
      embeddings: List[List[float]]
      good_indices: indices of documents that embedded successfully
    """
    all_embeddings: List[List[float]] = []

    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        batch_embeddings = embed_batch(batch, embedding_model)
        all_embeddings.extend(batch_embeddings)

    # Filter out failures (empty embeddings)
    good_indices = [
        idx for idx, emb in enumerate(all_embeddings) if emb
    ]

    embeddings_final = [
        all_embeddings[idx] for idx in good_indices
    ]

    return embeddings_final, good_indices


def upsert_chunks(
    collection,
    documents: List[str],
    metadatas: List[dict],
    ids: List[str],
    embeddings: List[List[float]],
):
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings,
    )


#--- MAIN INDEXING LOGIC ---
def index_project():
    collection = get_chunks_collection()

    documents, metadatas, ids = collect_chunks(
        root_dir=ROOT_DIR,
        project_id=PROJECT_ID,
    )

    if not documents:
        print("No chunks found.")
        return

    embeddings, good_indices = embed_documents(
        documents=documents,
        embedding_model=embedding_model,
        batch_size=8,
    )

    if not good_indices:
        print("No embeddings produced; aborting indexing.")
        return

    documents = [documents[i] for i in good_indices]
    metadatas = [metadatas[i] for i in good_indices]
    ids = [ids[i] for i in good_indices]

    upsert_chunks(
        collection=collection,
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings,
    )

    print(f"Indexed {len(documents)} chunks.")


if __name__ == "__main__":
    index_project()