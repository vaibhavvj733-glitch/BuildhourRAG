"""Save all chunks to a readable .txt file for inspection."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "ingestion"))

from loader import load_documents  # noqa: E402
from chunker import chunk_documents  # noqa: E402

DATA_DIR = _ROOT / "data"
OUTPUT = _ROOT / "chunks.txt"

files = sorted(p for p in DATA_DIR.glob("*") if p.is_file())
docs = load_documents(files)
chunks = chunk_documents(docs)

with open(OUTPUT, "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        source = Path(chunk.metadata.get("source", "unknown")).name
        f.write(f"=== Chunk {i} | source: {source} ===\n")
        f.write(chunk.page_content + "\n\n")

print(f"Wrote {len(chunks)} chunks to {OUTPUT}")
