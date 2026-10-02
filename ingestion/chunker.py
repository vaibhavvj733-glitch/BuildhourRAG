"""Document chunking utilities for Phase 1."""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Split documents into chunks with overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


if __name__ == "__main__":
    from loader import load_documents

    data_dir = Path(__file__).resolve().parent.parent / "data"
    files = sorted(p for p in data_dir.glob("*") if p.is_file())

    docs = load_documents(files)
    chunks = chunk_documents(docs)

    print(f"\nTotal documents loaded: {len(docs)}")
    print(f"Total chunks created: {len(chunks)}")

    if chunks:
        sample = chunks[0]
        print("\nSample chunk metadata:", sample.metadata)
        print("Sample chunk text (first 200 chars):")
        print(sample.page_content[:200])
