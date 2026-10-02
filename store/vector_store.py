"""Embeddings & vector store utilities for Phase 2."""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_STORE_PATH = Path(__file__).resolve().parent.parent / "vectorstore"


def get_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> FastEmbedEmbeddings:
    """Return a local embedding model (no API key required)."""
    return FastEmbedEmbeddings(model_name=model_name)


def build_vector_store(
    chunks: List[Document],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> FAISS:
    """Embed chunks and build a FAISS vector store.

    Adds a chunk_index to each chunk's metadata for citations.
    """
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    embeddings = get_embeddings(model_name)
    return FAISS.from_documents(chunks, embeddings)


def save_vector_store(vs: FAISS, path: str | Path = DEFAULT_STORE_PATH) -> None:
    """Persist the FAISS index to disk."""
    vs.save_local(str(path))
    print(f"Vector store saved to: {path}")


def load_vector_store(
    path: str | Path = DEFAULT_STORE_PATH,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> FAISS:
    """Load a previously saved FAISS index from disk."""
    embeddings = get_embeddings(model_name)
    return FAISS.load_local(
        str(path), embeddings, allow_dangerous_deserialization=True
    )


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ingestion"))

    from loader import load_documents
    from chunker import chunk_documents

    data_dir = Path(__file__).resolve().parent.parent / "data"
    files = sorted(p for p in data_dir.glob("*") if p.is_file())

    docs = load_documents(files)
    chunks = chunk_documents(docs)
    print(f"Chunks to embed: {len(chunks)}")

    vs = build_vector_store(chunks)
    save_vector_store(vs)

    vs_loaded = load_vector_store()
    print(f"Reloaded vector store with {vs_loaded.index.ntotal} vectors")
