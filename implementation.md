# implementation.md — RAG Chatbot (Phase-wise Build Guide)

This document breaks the implementation of the RAG chatbot into ordered phases. Each phase is self-contained and testable. Use [architecture.md](architecture.md) for component design and [PRD.md](PRD.md) for requirements.

**How to use with Cursor:** Implement one phase at a time. Paste the phase's prompt/instructions into Cursor, verify the phase's "Done when" criteria, then move to the next phase.

---

## Phase 0 — Project Setup

**Goal:** Create a runnable skeleton project.

**Tasks:**
- Create the project structure from architecture.md §5:
  ```
  rag-chatbot/
  ├── app.py
  ├── requirements.txt
  ├── .env
  ├── ingestion/ ├── store/ ├── retrieval/ ├── generation/ ├── ui/
  └── data/
  ```
- Write `requirements.txt` with: `streamlit`, `langchain`, `langchain-community`, `langchain-openai` (or `google-generativeai`/`ollama`), `faiss-cpu`, `pypdf`, `python-dotenv`.
- Create `.env.example` with `LLM_API_KEY=` and `MODEL_NAME=` placeholders. Load env vars via `python-dotenv`.
- Add a minimal `app.py` that runs a blank Streamlit page.
- Add a small sample document (e.g., `data/sample.md`) for testing.

**Done when:** `pip install -r requirements.txt` succeeds and `streamlit run app.py` shows a blank page.

---

## Phase 1 — Document Ingestion

**Goal:** Load PDF/TXT/MD files and split them into overlapping chunks.

**Tasks:**
- Implement `ingestion/loader.py`:
  - `load_documents(paths) -> list[Document]` using `PyPDFLoader`, `TextLoader`, and markdown handling via `UnstructuredMarkdownLoader` or plain `TextLoader`.
  - Skip unsupported/corrupt files with a printed warning.
- Implement `ingestion/chunker.py`:
  - `chunk_documents(docs, chunk_size=1000, chunk_overlap=200) -> list[Document]` using `RecursiveCharacterTextSplitter`.
- Add `__main__` test block or a small script to load files from `data/` and print chunk counts and a sample chunk.

**Done when:** Running the ingestion module on sample files prints document counts, chunk counts, and chunk metadata (source, page).

---

## Phase 2 — Embeddings & Vector Store

**Goal:** Embed chunks and persist them in FAISS/Chroma.

**Tasks:**
- Implement `store/vector_store.py`:
  - `build_vector_store(chunks)` → creates embeddings (OpenAI `text-embedding-3-small`, or local `sentence-transformers`/Ollama embeddings) and a FAISS index.
  - `save_vector_store(vs, path="vectorstore/")` and `load_vector_store(path)`.
- Keep chunk metadata (source file, chunk index) attached for citations later.
- Add a test script: build the store from sample docs, save it, reload it.

**Done when:** A `vectorstore/` directory is created and can be reloaded without re-embedding.

---

## Phase 3 — Retrieval

**Goal:** Retrieve top-k relevant chunks for a query.

**Tasks:**
- Implement `retrieval/retriever.py`:
  - `get_retriever(vector_store, k=4)` → LangChain retriever or a manual `similarity_search(query, k)`.
  - Return chunks with their similarity scores.
- Test script: run 2–3 sample queries and print the top-k chunk sources and snippets.

**Done when:** Queries return sensible chunks whose text relates to the query.

---

## Phase 4 — Generation (LLM Answer with Citations)

**Goal:** Generate grounded answers from retrieved chunks.

**Tasks:**
- Implement `generation/llm.py`:
  - Build a prompt template: system instruction ("Answer only from the context; if unknown, say so") + context chunks + question.
  - `generate_answer(query, chunks) -> str` calls the LLM (OpenAI / Gemini / Ollama per `.env`).
  - Append source citations: document name + chunk index/page.
- Test script: full pipeline — load → chunk → embed → retrieve → answer — on a sample question.

**Done when:** A sample question returns a concise answer citing source documents.

---

## Phase 5 — Chat Interface

**Goal:** Wire everything into a Streamlit UI.

**Tasks:**
- Implement `app.py` with Streamlit:
  - Sidebar: document upload (PDF/TXT/MD) or select from `data/`, "Build Index" button, model info.
  - On index build: run ingestion → chunking → embedding → save vector store.
  - Chat area: `st.chat_input`, message history via `st.session_state`, display answers.
  - Under each answer, an expander showing retrieved chunks and their sources/scores.
  - "Clear chat" button to reset conversation history.
- Cache the vector store across reruns (`st.cache_resource`).

**Done when:** User can upload docs, build an index, ask questions, see cited answers, expand sources, and clear the chat.

---

## Phase 6 — Polish & Demo Readiness

**Goal:** Meet the PRD's non-functional and success criteria.

**Tasks:**
- Response-time check: keep top-k small (3–5), cache the embedding model and vector store.
- Graceful error handling: no docs indexed, empty retrieval, LLM API errors → friendly UI messages.
- One-command run: document `streamlit run app.py` in a `README.md`.
- Demo script: prepare 3–5 example questions that showcase citations.
- Verify success criteria from PRD §10: answers grounded in docs, citations shown, pipeline explainable step-by-step.

**Done when:** The app runs locally with one command and reliably answers demo questions with citations.

---

## Phase Order & Dependencies

```
Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6
```

Each phase only depends on the previous ones. Do not skip ahead: the UI (Phase 5) should be a thin layer over the tested pipeline modules from Phases 1–4.

## Tips for Guiding Cursor

- Paste one phase at a time; include the relevant section of `architecture.md` for context.
- After each phase, run the phase's test script before proceeding.
- Ask Cursor to keep modules in the exact files/folders listed in `architecture.md` §5.
