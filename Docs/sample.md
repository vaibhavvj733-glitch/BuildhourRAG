# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation is a technique that combines document retrieval with large language model generation. Instead of relying only on the model's parametric knowledge, a RAG system first retrieves relevant passages from a knowledge base and provides them as context to the LLM.

## Why RAG?

Traditional LLMs can hallucinate or lack up-to-date domain-specific knowledge. RAG addresses this by grounding answers in the actual source documents. The model is instructed to answer only from the retrieved context, which improves factual accuracy and allows source citations.

## The RAG Pipeline

1. **Ingestion**: Load documents (PDF, TXT, MD) from disk.
2. **Chunking**: Split documents into smaller chunks with overlap, so context is preserved across boundaries.
3. **Embedding**: Convert each chunk into a dense vector using an embedding model.
4. **Indexing**: Store the vectors in a vector store such as FAISS or Chroma.
5. **Retrieval**: Embed the user query and find the top-k most similar chunks.
6. **Generation**: Send the retrieved chunks and query to the LLM to produce a grounded answer.

## Chunking Strategies

Chunking is critical for retrieval quality. Chunks that are too large dilute relevance; chunks that are too small lose context. A common starting point is 1000 characters with 200 characters of overlap.

## Evaluation

A working RAG demo should answer questions using only the provided documents, cite the source documents, and admit when the answer is not available in the knowledge base.
