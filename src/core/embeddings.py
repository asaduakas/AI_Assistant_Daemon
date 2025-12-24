import numpy as np
from src.config import settings
from typing import Any, List, Tuple

def assert_embedding_dim(vec: List[float]) -> None:
    if len(vec) != settings.EMBED_DIM:
        raise ValueError(f"Embedding dim mismatch: expected {settings.EMBED_DIM}, got {len(vec)}")

def mean_pool(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        raise ValueError("vectors must not be empty")

    arr = np.asarray(vectors, dtype=float)

    if arr.ndim != 2:
        raise ValueError("vectors must be a 2D array")

    return arr.mean(axis=0).tolist()

def normalize_embeddings(res: Any) -> List[List[float]]:
    """
    Normalize llama.cpp embedding outputs into:
        List[List[float]]  # one vector per document
    """
    emb = res[0] if isinstance(res, tuple) else res

    # Convert numpy arrays
    try:
        import numpy as np
        if isinstance(emb, np.ndarray):
            emb = emb.tolist()
    except Exception:
        pass

    if not emb:
        raise ValueError("Empty embedding result")

    # Case 1: single vector -> one doc
    if isinstance(emb[0], (int, float)):
        return [emb]

    # Case 2: token embeddings for ONE document
    # shape: tokens × dim
    if (
        isinstance(emb[0], list)
        and emb
        and isinstance(emb[0][0], (int, float))
    ):
        return [mean_pool(emb)]

    # Case 3: batched token embeddings
    # shape: docs × tokens × dim
    if (
        isinstance(emb[0], list)
        and emb
        and isinstance(emb[0][0], list)
        and isinstance(emb[0][0][0], (int, float))
    ):
        return [mean_pool(doc_tokens) for doc_tokens in emb]

    raise ValueError(
        f"Unknown embedding structure: "
        f"{type(emb)} / sample={type(emb[0])}"
    )


def embed_one_document(doc: str, embedding_model: Any,) -> List[float]:
    """
    Embed a single document.

    Raises if embedding fails.
    """
    res = embedding_model.embed(doc)
    batch = normalize_embeddings(res)
    vec = batch[0]
    assert_embedding_dim(vec)

    return vec

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

        for vec in embeddings:
            assert_embedding_dim(vec)

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


def embed_documents(documents: List[str], embedding_model: Any, batch_size: int = 1,
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