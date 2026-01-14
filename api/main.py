from fastapi import FastAPI, Query
from rag import retriever
from rag.fusion import rrf
from rag.reranker import BGEReranker
from rag.llm_model import GeminiModel

app = FastAPI(title="Finance RAG API")

# Initialize
reranker = BGEReranker()
llm = GeminiModel()

@app.on_event("startup")
def startup_event():
    retriever.load_faiss("artifacts/faiss_index/index.faiss",
                         "artifacts/faiss_index/meta.pkl")
    print("✅ FAISS index + meta loaded")

@app.get("/ask")
async def ask(q: str = Query(..., description="User query"), k: int = 10):
    # Hybrid retrieval
    bm25_res, dense_res = retriever.dual(q, k)
    if not bm25_res and not dense_res:
        return {"query": q, "answer": "No relevant documents found.", "chunks_used": 0}

    # RRF Fusion
    bm25_rank = {d["chunk_id"]: i for i, d in enumerate(bm25_res)}
    dense_rank = {d["chunk_id"]: i for i, d in enumerate(dense_res)}
    fused = rrf([bm25_rank, dense_rank])
    fused_ids = [doc_id for doc_id, _ in fused[:k]]

    doc_map = {d["chunk_id"]: d for d in bm25_res + dense_res}
    docs = [doc_map[cid] for cid in fused_ids if cid in doc_map]

    if not docs:
        return {"query": q, "answer": "Fusion produced no usable documents.", "chunks_used": 0}

    # Rerank
    reranked_docs = reranker.cross_encode(q, docs, topn=5)

    # Generate answer using Gemini
    answer = await llm.generate(q, reranked_docs)


    return {"query": q, "answer": answer, "chunks_used": len(reranked_docs)}
