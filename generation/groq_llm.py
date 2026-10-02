"""Groq-backed answer generation (used by the UI)."""

import os
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "generation"))

load_dotenv(_ROOT / ".env")

from llm import build_prompt  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

GROQ_MODEL = os.getenv("MODEL_NAME") or "qwen/qwen3.8-27b"

SYSTEM_RULES = (
    "You answer factual questions about mutual funds using ONLY the provided context. "
    "Keep answers to at most 3 sentences. "
    "Do NOT give investment advice; if the user asks for opinions (should I buy/sell), "
    "politely refuse and say this assistant provides facts only. "
    "If the answer is not in the context, say \"I don't know based on the provided documents.\""
)


def generate_answer_groq(query: str, chunks: List[Document]) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in .env")

    if not chunks:
        return "I don't know based on the provided documents.\n\nSources: none"

    prompt = build_prompt(query, chunks)

    from groq import Groq

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_RULES},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=300,
    )

    answer = response.choices[0].message.content.strip()

    sources = []
    seen = set()
    for chunk in chunks:
        source = Path(chunk.metadata.get("source", "unknown")).name
        idx = chunk.metadata.get("chunk_index", "?")
        if (source, idx) not in seen:
            seen.add((source, idx))
            sources.append(f"{source} (chunk {idx})")

    return f"{answer}\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)
