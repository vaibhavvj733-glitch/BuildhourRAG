# architecture.md — RAG Chatbot (Class Demo)

## 1. Overview

This document describes the architecture of the RAG (Retrieval-Augmented Generation) chatbot defined in [PRD.md](PRD.md). The system answers natural-language questions by retrieving relevant chunks from a document knowledge base and generating grounded answers with source citations.

## 2. High-Level Architecture

```
┌─────────────┐   upload/select   ┌──────────────────────────────────────┐
│    User     │ ────────────────► │        Chat Interface (UI)           │
└─────────────┘ ◄──────────────── │   Streamlit / Gradio / CLI           │
       ▲            answer +       └──────────────┬───────────────────────┘
       │            citations                     │
       │                                          ▼
       │                          ┌──────────────────────────────────────┐
       └───────────────────────── │        Answer Generator             │
                  grounded answer │   LLM (OpenAI/Gemini/Ollama)        │
                                  └──────────────▲───────────────────────┘
                                                 │ prompt = query + top-k chunks
                                  ┌──────────────┴───────────────────────┐
                                  │        Retrieval Engine              │
                                  │   embed query → vector search → top-k│
                                  └──────────────▲───────────────────────┘
                                                 │
                                  ┌──────────────┴───────────────────────┐
                                  │        Vector Store (FAISS/Chroma)    │
                                  └──────────────▲───────────────────────┘
                                                 │ embeddings
┌────────────────────────────────────────────────┴───────────────────────┐
│                     Document Ingestion Pipeline (offline)             │
│   Load (PDF/TXT/MD) → Split into chunks (with overlap) → Embed        │
└────────────────────────────────────────────────────────────────────────┘
```

## 3. Components

### 3.1 Document Ingestion (`ingestion/`)
- **Loader**: Reads local files in PDF, TXT, and MD formats (e.g., `PyPDFLoader`, `TextLoader`).
- **Chunker**: Splits documents into fixed-size chunks with configurable overlap to preserve context across boundaries.

### 3.2 Embeddings & Vector Store (`store/`)
- **Embedding model**: Converts text chunks and queries into dense vectors.
- **Vector store**: FAISS (default, in-memory/local) or Chroma (persistent) storing chunk vectors + metadata (source file, chunk index).

### 3.3 Retrieval (`retrieval/`)
- Embeds the user query with the same embedding model.
- Performs similarity search and returns the top-k most relevant chunks with scores.

### 3.4 Generation (`generation/`)
- Builds a prompt containing the user query and retrieved chunks.
- Calls the LLM to produce a concise, grounded answer.
- Appends source references (document name, chunk index) to the response.

### 3.5 Chat Interface (`ui/`)
- Streamlit or Gradio app providing:
  - Document upload / selection
  - Chat input and conversation display
  - Expandable "retrieved context / sources" section
  - Clear/new conversation button

## 4. Data Flow

### 4.1 Indexing (one-time / on document change)
1. User selects documents.
2. Documents are loaded and split into chunks.
3. Each chunk is embedded and stored in the vector store with metadata.

### 4.2 Querying (per question)
1. User enters a question.
2. Query is embedded; top-k chunks retrieved from the vector store.
3. Retrieved chunks + query are sent to the LLM.
4. LLM returns an answer; UI displays answer, sources, and retrieved context.

## 5. Proposed Project Structure

```
rag-chatbot/
├── app.py                  # Entry point (Streamlit/Gradio UI)
├── requirements.txt
├── .env                    # API keys (LLM provider)
├── ingestion/
│   ├── loader.py           # PDF/TXT/MD loading
│   └── chunker.py          # Chunking with overlap
├── store/
│   └── vector_store.py     # FAISS/Chroma setup & persistence
├── retrieval/
│   └── retriever.py        # Top-k similarity search
├── generation/
│   └── llm.py              # Prompt construction + LLM call
├── ui/
│   └── chat.py             # Chat interface components
└── data/                   # Sample documents for the demo
```

## 6. Technology Choices

| Concern | Choice | Rationale |
|---|---|---|
| Language | Python | PRD-suggested; rich RAG ecosystem |
| Orchestration | LangChain (or direct API) | Simplifies pipeline composition |
| LLM | OpenAI / Gemini / Ollama | Demo-friendly; Ollama for offline use |
| Vector store | FAISS (default) / Chroma | Lightweight, local, no server needed |
| UI | Streamlit | Fastest to build a demo UI |

## 7. Configuration

- Chunk size, chunk overlap, top-k, and model names are configurable via constants or a small config file.
- API keys are read from environment variables (`.env`), never hard-coded.

## 8. Error Handling & Edge Cases

- Unsupported/corrupt files are skipped with a warning.
- Empty retrieval results prompt the LLM to say it doesn't know based on the provided documents.
- Long documents are chunked; very large PDFs may be capped for demo speed.

## 9. Non-Functional Design Decisions

- **Performance**: In-memory FAISS and small top-k (e.g., 3–5) keep responses within a few seconds.
- **Simplicity**: Single-command startup (e.g., `streamlit run app.py`).
- **Modularity**: Ingestion, retrieval, and generation are separate modules, matching the PRD's separation requirement.

## 10. Future Extensions (Out of Scope for Demo)

- Reranking and hybrid (keyword + vector) search
- Persistent chat history / multi-session support
- Web deployment and authentication
