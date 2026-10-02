"""Answer generation with citations for Phase 4."""

import sys
from pathlib import Path
from typing import List

from langchain_core.documents import Document

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "ingestion"))
sys.path.insert(0, str(_ROOT / "store"))
sys.path.insert(0, str(_ROOT / "retrieval"))

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."
Keep the answer concise.

Context:
{context}

Question: {question}

Answer:"""

_pipeline = None


def _get_pipeline(model_name: str = DEFAULT_MODEL):
    """Lazily load the local text-generation pipeline."""
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline

        _pipeline = pipeline(
            "text-generation",
            model=model_name,
            max_new_tokens=150,
            do_sample=False,
        )
    return _pipeline


def build_prompt(query: str, chunks: List[Document]) -> str:
    """Format retrieved chunks and query into the prompt."""
    context = "\n\n".join(
        f"[{i + 1}] {chunk.page_content}" for i, chunk in enumerate(chunks)
    )
    return PROMPT_TEMPLATE.format(context=context, question=query)


def generate_answer(query: str, chunks: List[Document]) -> str:
    """Generate a grounded answer and append source citations."""
    if not chunks:
        return "I don't know based on the provided documents.\n\nSources: none"

    prompt = build_prompt(query, chunks)
    pipe = _get_pipeline()
    output = pipe(prompt)[0]["generated_text"]

    # Strip the prompt from the generated text
    answer = output[len(prompt):].strip()
    if not answer:
        answer = "I don't know based on the provided documents."

    citations = []
    seen = set()
    for chunk in chunks:
        source = Path(chunk.metadata.get("source", "unknown")).name
        idx = chunk.metadata.get("chunk_index", "?")
        key = (source, idx)
        if key not in seen:
            seen.add(key)
            citations.append(f"- {source} (chunk {idx})")

    return f"{answer}\n\nSources:\n" + "\n".join(citations)


if __name__ == "__main__":
    from loader import load_documents
    from chunker import chunk_documents
    from vector_store import build_vector_store, save_vector_store, load_vector_store, DEFAULT_STORE_PATH
    from retriever import retrieve_with_scores

    data_dir = _ROOT / "data"
    files = sorted(p for p in data_dir.glob("*") if p.is_file())

    if DEFAULT_STORE_PATH.exists():
        vs = load_vector_store()
    else:
        docs = load_documents(files)
        chunks = chunk_documents(docs)
        vs = build_vector_store(chunks)
        save_vector_store(vs)

    query = "What is RAG and why is it useful?"
    print(f"Query: {query}\n")

    results = retrieve_with_scores(vs, query, k=2)
    chunks = [doc for doc, _ in results]

    print(generate_answer(query, chunks))
