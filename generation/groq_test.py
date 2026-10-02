"""Test the RAG pipeline with Groq before building the UI.

Usage:
    python generation/groq_test.py "Your question here"
    python generation/groq_test.py        (then type the question)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "store"))
sys.path.insert(0, str(_ROOT / "retrieval"))
sys.path.insert(0, str(_ROOT / "generation"))

load_dotenv(_ROOT / ".env")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from vector_store import load_vector_store  # noqa: E402
from retriever import retrieve_with_scores  # noqa: E402
from llm import build_prompt  # noqa: E402

GROQ_MODEL = os.getenv("MODEL_NAME") or "openai/gpt-oss-20b"


def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY is not set in .env")
        sys.exit(1)

    if len(sys.argv) > 1:
        questions = [" ".join(sys.argv[1:])]
    else:
        print("Type your question and press Enter. Type 'exit' to quit.\n")
        questions = None

    vs = load_vector_store()
    from groq import Groq

    client = Groq(api_key=api_key)

    while True:
        query = questions.pop(0) if questions else input("You: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break
        if questions is not None and not questions:
            pass

        results = retrieve_with_scores(vs, query, k=2)
        chunks = [doc for doc, _ in results]

        print(f"\nRetrieved {len(chunks)} chunks:")
        for doc, score in results:
            source = Path(doc.metadata.get("source", "unknown")).name
            print(f"  score={score:.4f} | {source} | chunk {doc.metadata.get('chunk_index')}")

        prompt = build_prompt(query, chunks)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
        )

        answer = response.choices[0].message.content.strip()
        print(f"\nAnswer (model: {GROQ_MODEL}):\n{answer}")

        print("\nSources:")
        seen = set()
        for chunk in chunks:
            source = Path(chunk.metadata.get("source", "unknown")).name
            idx = chunk.metadata.get("chunk_index", "?")
            if (source, idx) not in seen:
                seen.add((source, idx))
                print(f"  - {source} (chunk {idx})")
        print()

        if questions is not None:
            break


if __name__ == "__main__":
    main()
