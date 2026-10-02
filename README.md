# Mutual Fund FAQ Assistant — RAG Chatbot (Class Demo)

A Retrieval-Augmented Generation (RAG) chatbot that answers factual questions about 5 HDFC mutual fund schemes using only the provided source data.

## Scope
- AMC: HDFC Mutual Fund
- Schemes: Large Cap, Flexi Cap, ELSS, Small Cap, Balanced Advantage (Direct, Growth)
- Facts-only answers with one source link; no investment advice

## Setup
```powershell
pip install -r requirements.txt
# add GROQ_API_KEY to .env
python store/vector_store.py        # build the vector index once
```

## Run
```powershell
python -m streamlit run app.py
```

## Pipeline
- Ingestion: Load (`data/*.md`) → Chunk (1000 chars, 200 overlap) → Embed (FastEmbed, 384-dim) → Store (FAISS in `vectorstore/`)
- Query: Question → Embed → Retrieve top 2 chunks → Groq LLM (`openai/gpt-oss-20b`) → Answer + sources

## Inspectable artifacts
- `chunks.txt` — all chunks in readable text (`python ingestion/dump_chunks.py`)
- `embeddings.txt` — chunk embeddings (`python store/dump_embeddings.py`)
- `retrieval/query.py "..."` — test retrieval on a query
- `generation/groq_test.py` — interactive Q&A without the UI

## Known limits
- Embedding model is FastEmbed `BAAI/bge-small-en-v1.5` (brief asked for sentence-transformers MiniLM, which fails to import on this machine due to a blocked scipy DLL)
- Vector DB is FAISS persisted to disk (brief suggested ChromaDB)
- No performance/return computations; historical returns are not compared
- No PII is accepted or stored
