Hybrid RAG Architecture: Combines keyword search (BM25) with vector similarity search (FAISS) and rank fusion for ~35% improvement in Top-5 document retrieval precision over vector-only search.

AI-Powered Financial Analysis: Extracts actionable insights from financial reports, RBI datasets, and market data using GenAI.

Asynchronous FastAPI Service: Handles concurrent user queries with ~1.5s p95 end-to-end latency across retrieval, fusion, and generation.

Production-Grade Scalability: Fully containerized using Docker

Custom Embeddings: Uses BAAI/bge-base-en-v1.5 for semantic embeddings of financial documents.

Extensible: Modular architecture allows addition of new datasets, models, or retrieval strategies.

 Tech Stack

AI/ML: LlamaParse, LlamaCpp, BAAI embeddings, GenAI gemini models

Data & Storage:  FAISS, Pickle, Pandas, NumPy

Search & Retrieval: BM25, FAISS vector search, Reciprocal Rank Fusion

Backend & API: FastAPI, Async processing, Uvicorn

Orchestration & Deployment: Docker

Programming Languages: Python
