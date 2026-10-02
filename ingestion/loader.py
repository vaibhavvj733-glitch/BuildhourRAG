"""Document loading utilities for Phase 1."""

from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_documents(paths: Iterable[str | Path]) -> List[Document]:
    """Load documents from local files (PDF, TXT, MD).

    Unsupported or corrupt files are skipped with a printed warning.
    """
    documents: List[Document] = []

    for path in paths:
        path = Path(path)

        if not path.exists():
            print(f"Warning: file not found, skipping: {path}")
            continue

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            print(f"Warning: unsupported file type '{ext}', skipping: {path}")
            continue

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(str(path))
            else:  # .txt or .md
                loader = TextLoader(str(path), encoding="utf-8")

            documents.extend(loader.load())
            print(f"Loaded: {path}")
        except Exception as exc:
            print(f"Warning: failed to load {path}: {exc}")

    return documents


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"
    files = sorted(p for p in data_dir.glob("*") if p.is_file())
    docs = load_documents(files)
    print(f"\nTotal documents loaded: {len(docs)}")
