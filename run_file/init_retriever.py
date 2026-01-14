# run_file/init_retriever.py
from rag.retriever import load_faiss

load_faiss(
    "artifacts/faiss_index/index.faiss",
    "artifacts/faiss_index/meta.pkl"
)

print("FAISS retriever initialized")
