"""Retrieval utilities for Phase 3."""

import sys
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "store"))

from vector_store import load_vector_store  # noqa: E402


def get_retriever(vector_store: FAISS, k: int = 4):
    """Return a LangChain retriever over the vector store."""
    return vector_store.as_retriever(search_kwargs={"k": k})


def retrieve_with_scores(
    vector_store: FAISS, query: str, k: int = 4
) -> List[Tuple[Document, float]]:
    """Return the top-k chunks with their similarity scores."""
    return vector_store.similarity_search_with_score(query, k=k)


if __name__ == "__main__":
    vs = load_vector_store()

    sample_queries = [
        "What is RAG?",
        "How does the RAG pipeline work?",
        "What chunk size should I use?",
    ]

    for query in sample_queries:
        print(f"\nQuery: {query}")
        results = retrieve_with_scores(vs, query, k=2)
        for doc, score in results:
            source = Path(doc.metadata.get("source", "unknown")).name
            print(f"  score={score:.4f} | source={source} | chunk={doc.metadata.get('chunk_index')}")
            print(f"    {doc.page_content[:120]!r}")
