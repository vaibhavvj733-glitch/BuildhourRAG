"""Streamlit chat interface for the RAG chatbot (Phase 5)."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ingestion"))
sys.path.insert(0, str(ROOT / "store"))
sys.path.insert(0, str(ROOT / "retrieval"))
sys.path.insert(0, str(ROOT / "generation"))

from loader import load_documents  # noqa: E402
from chunker import chunk_documents  # noqa: E402
from vector_store import (  # noqa: E402
    build_vector_store,
    save_vector_store,
    load_vector_store,
    DEFAULT_STORE_PATH,
)
from retriever import retrieve_with_scores  # noqa: E402
from groq_llm import generate_answer_groq  # noqa: E402


def get_answer(query, chunks):
    """Groq first; fall back to the local model if Groq is unavailable."""
    try:
        return generate_answer_groq(query, chunks)
    except Exception as exc:
        from llm import generate_answer

        st.warning(f"Groq unavailable ({exc}); using local model.")
        return generate_answer(query, chunks)


def source_link(doc):
    """Extract the Source: URL from chunk text if present."""
    for line in doc.page_content.splitlines():
        if line.strip().lower().startswith("source:"):
            return line.split(":", 1)[1].strip()
    return None

DATA_DIR = ROOT / "data"
UPLOAD_DIR = ROOT / "data" / "uploads"


@st.cache_resource
def get_vector_store():
    if DEFAULT_STORE_PATH.exists():
        return load_vector_store()
    # Auto-build from data/ on first run (e.g., fresh Render deploy)
    files = sorted(p for p in DATA_DIR.glob("*") if p.is_file() and p.parent == DATA_DIR and p.name != "uploads")
    if not files:
        return None
    docs = load_documents(files)
    if not docs:
        return None
    chunks = chunk_documents(docs)
    vs = build_vector_store(chunks)
    save_vector_store(vs)
    return vs


def build_index_from_paths(paths):
    docs = load_documents(paths)
    if not docs:
        st.warning("No documents could be loaded.")
        return None
    chunks = chunk_documents(docs)
    vs = build_vector_store(chunks)
    save_vector_store(vs)
    st.cache_resource.clear()
    return vs


def main():
    st.set_page_config(page_title="Mutual Fund FAQ Assistant", page_icon="💬", layout="wide")

    st.markdown(
        """
        <style>
        .hero {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 1.2rem;
        }
        .hero h1 { color: #f5f7fa; margin: 0; font-size: 2.1rem; }
        .hero p { color: #c3cfe2; margin: 0.5rem 0 0 0; }
        .badge {
            display: inline-block; background: #f7971e; color: #1b1b1b;
            padding: 0.2rem 0.8rem; border-radius: 20px; font-weight: 600;
            font-size: 0.8rem; margin-bottom: 0.6rem;
        }
        .chip {
            display: inline-block; background: #eef2f7; border: 1px solid #d5deea;
            border-radius: 20px; padding: 0.3rem 0.9rem; margin: 0.2rem;
            font-size: 0.85rem; color: #2c5364;
        }
        .answer-card {
            border-left: 4px solid #2c5364; background: #f7fafc;
            padding: 1rem 1.2rem; border-radius: 8px;
        }
        </style>
        <div class="hero">
            <span class="badge">FACTS-ONLY • NO ADVICE</span>
            <h1>💬 Mutual Fund FAQ Assistant</h1>
            <p>Ask factual questions about HDFC schemes — expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer, benchmark.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "**Try asking:** "
        "<span class='chip'>What is the expense ratio of HDFC Large Cap?</span>"
        "<span class='chip'>What is the ELSS lock-in period?</span>"
        "<span class='chip'>What is the minimum SIP for HDFC Small Cap?</span>",
        unsafe_allow_html=True,
    )
    st.write("")

    with st.sidebar:
        st.markdown("### 🗂️ Knowledge Base")

        uploaded = st.file_uploader(
            "Upload documents (PDF, TXT, MD)",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        if uploaded and st.button("Index uploaded files"):
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            paths = []
            for f in uploaded:
                p = UPLOAD_DIR / f.name
                p.write_bytes(f.getbuffer())
                paths.append(p)
            with st.spinner("Building index..."):
                build_index_from_paths(paths)
            st.success("Index built from uploaded files.")

        data_files = sorted(p for p in DATA_DIR.glob("*") if p.is_file())
        if data_files and st.button("Index files in data/"):
            with st.spinner("Building index..."):
                build_index_from_paths(data_files)
            st.success("Index built from data/ files.")

        st.caption(f"Index path: {DEFAULT_STORE_PATH}")

        if st.button("🗑️ Clear chat"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("context"):
                with st.expander("Retrieved context / sources"):
                    for item in msg["context"]:
                        st.markdown(
                            f"**{item['source']}** (chunk {item['chunk_index']}, score {item['score']:.3f})"
                        )
                        st.caption(item["text"][:300])
                        if item.get("link"):
                            st.markdown(f"🔗 [Source]({item['link']})")

    query = st.chat_input("Ask a question about your documents...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        vs = get_vector_store()
        with st.chat_message("assistant"):
            if vs is None:
                msg = "No index found. Please build the index from the sidebar first."
                st.markdown(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
            else:
                with st.spinner("Thinking..."):
                    results = retrieve_with_scores(vs, query, k=2)
                    chunks = [doc for doc, _ in results]
                    answer = get_answer(query, chunks)
                st.markdown('<div class="answer-card">', unsafe_allow_html=True)
                st.markdown(answer)
                st.markdown('</div>', unsafe_allow_html=True)
                context = [
                    {
                        "source": Path(doc.metadata.get("source", "unknown")).name,
                        "chunk_index": doc.metadata.get("chunk_index", "?"),
                        "score": float(score),
                        "text": doc.page_content,
                        "link": source_link(doc),
                    }
                    for doc, score in results
                ]
                with st.expander("Retrieved context / sources"):
                    for item in context:
                        st.markdown(
                            f"**{item['source']}** (chunk {item['chunk_index']}, score {item['score']:.3f})"
                        )
                        st.caption(item["text"][:300])
                        if item.get("link"):
                            st.markdown(f"🔗 [Source]({item['link']})")
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "context": context}
                )


if __name__ == "__main__":
    main()
