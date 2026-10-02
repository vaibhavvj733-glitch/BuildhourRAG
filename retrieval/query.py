"""Ask a query and see which chunks are retrieved."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "store"))
sys.path.insert(0, str(_ROOT / "retrieval"))

from vector_store import load_vector_store  # noqa: E402
from retriever import retrieve_with_scores  # noqa: E402

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter query: ")
    k = 2

    vs = load_vector_store()
    results = retrieve_with_scores(vs, query, k=k)

    print(f"\nQuery: {query}\nTop {k} results:")
    for doc, score in results:
        source = Path(doc.metadata.get("source", "unknown")).name
        print(f"\n  score={score:.4f} | {source} | chunk {doc.metadata.get('chunk_index')}")
        print(f"  {doc.page_content[:200]!r}")
