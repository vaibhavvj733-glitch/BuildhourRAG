# PRD.md — RAG Chatbot (Class Demo)

## 1. Overview

A Retrieval-Augmented Generation (RAG) chatbot built as a class demo. The chatbot answers user questions by retrieving relevant documents from a knowledge base and using an LLM to generate grounded answers with citations.

## 2. Problem Statement

Users need quick, accurate answers from a specific set of documents. Traditional LLMs may hallucinate or lack up-to-date, domain-specific knowledge. A RAG pipeline solves this by combining document retrieval with LLM generation.

## 3. Goals

- Demonstrate the end-to-end RAG pipeline: document ingestion → chunking → embedding → vector search → LLM generation.
- Provide accurate answers grounded in the source documents.
- Show source citations / retrieved context to the user.
- Keep the scope small and demo-friendly.

## 4. Non-Goals

- Production-grade scaling, authentication, or multi-tenancy.
- Fine-tuning models.
- Supporting every document format (start with common formats like PDF/TXT/MD).

## 5. Target Users

- Classmates / instructor evaluating the demo.
- Students learning about RAG architectures.

## 6. User Stories

- As a user, I can upload or select a set of documents for the chatbot to use as its knowledge base.
- As a user, I can ask questions in natural language and get answers based on the documents.
- As a user, I can see which parts of the documents were used to answer my question.
- As a user, I can start a new chat / clear conversation history.

## 7. Functional Requirements

### 7.1 Document Ingestion
- Load documents from local files (PDF, TXT, MD).
- Split documents into chunks with overlap.

### 7.2 Embeddings & Storage
- Generate embeddings for each chunk.
- Store embeddings in a vector store (e.g., FAISS, Chroma).

### 7.3 Retrieval
- Given a user query, retrieve the top-k most relevant chunks.

### 7.4 Generation
- Send the retrieved chunks + query to an LLM.
- Return a concise answer with references to source chunks.

### 7.5 Chat Interface
- Simple UI (CLI, Streamlit, or Gradio) for asking questions.
- Display answer and retrieved context/sources.

## 8. Non-Functional Requirements

- Response time: answer within a few seconds for demo purposes.
- Simple to run locally with a single command.
- Clear separation of pipeline components (ingestion, retrieval, generation).

## 9. Tech Stack (Suggested)

- Python
- LangChain or direct OpenAI/Gemini/Ollama API
- FAISS or Chroma vector store
- Streamlit / Gradio for UI

## 10. Success Criteria

- The demo successfully answers questions using the provided documents.
- Answers cite the source documents.
- The pipeline can be explained step-by-step during the class presentation.

## 11. Out of Scope / Future Work

- More advanced retrieval (reranking, hybrid search).
- Persistent chat history.
- Deployment to the web.
