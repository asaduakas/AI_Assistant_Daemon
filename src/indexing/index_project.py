import ast, hashlib, logging
from pathlib import Path
from typing import Iterator, Tuple, List, Any
import numpy as np

from src.config import settings
from src.vectorstore.chroma_store import get_chunks_collection
from src.core.embeddings import embed_documents
from src.core.model_loader import get_embedding_model

#---helpers---
def iter_source_files(root: Path) -> Iterator[Path]:
    for item in settings.INCLUDE_DIRS:
        target = root / item

        if target.is_file() & (target.suffix in settings.SUPPORTED_EXTENSIONS):
            yield target

        elif target.is_dir():
            for path in target.rglob("*"):
                if path.is_file() and path.suffix in settings.SUPPORTED_EXTENSIONS:
                    yield path

def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def sha1(text:str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
    
def chunk_stats(text:str) -> dict:
    return{
        "chars": len(text),
        "lines": text.count("\n") + 1,
        "words": len(text.split())
    }


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

            ch_stats = chunk_stats(chunk_text)
            if ch_stats["chars"] > 6000:
                print(
                    f"[LARGE CHUNK] {relpath}:{start}-{end}"
                    f"chars={ch_stats['chars']} lines={ch_stats['lines']}"
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


#--- MAIN INDEXING LOOP ---
def index_project():
    collection = get_chunks_collection()

    documents, metadatas, ids = collect_chunks(
        root_dir=settings.ROOT_DIR,
        project_id=settings.PROJECT_ID,
    )

    if not documents:
        print("No chunks found.")
        return

    embeddings, good_indices = embed_documents(
        documents=documents,
        embedding_model=get_embedding_model(),
        batch_size=1,
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
    from src.utils.logger import setup_logging
    import logging
    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("Application started")
    index_project()