"""Dump chunk embeddings to a human-readable text file."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "ingestion"))
sys.path.insert(0, str(_ROOT / "store"))

from loader import load_documents  # noqa: E402
from chunker import chunk_documents  # noqa: E402
from vector_store import get_embeddings  # noqa: E402

DATA_DIR = _ROOT / "data"
OUTPUT = _ROOT / "embeddings.txt"


def main():
    files = sorted(p for p in DATA_DIR.glob("*") if p.is_file())
    docs = load_documents(files)
    chunks = chunk_documents(docs)

    embeddings_model = get_embeddings()
    vectors = embeddings_model.embed_documents([c.page_content for c in chunks])

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            source = Path(chunk.metadata.get("source", "unknown")).name
            f.write(f"=== Chunk {i} | source: {source} ===\n")
            f.write(f"Text: {chunk.page_content[:150]!r}...\n")
            f.write(f"Embedding dimension: {len(vector)}\n")
            f.write("Embedding vector:\n[")
            f.write(", ".join(f"{v:.6f}" for v in vector))
            f.write("]\n\n")

    print(f"Wrote {len(chunks)} embeddings to {OUTPUT}")


if __name__ == "__main__":
    main()
