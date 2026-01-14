import os
import time
import pickle
from rag.ingest import load_pdfs
from rag.chunking import chunk_text

OUTPUT_PATH = "artifacts/colab_chunks.pkl"

def main():
    os.makedirs("artifacts", exist_ok=True)

    start_time = time.time()

    docs = load_pdfs("data/raw_pdfs")
    total_docs = len(docs)

    print(f"📄 Found {total_docs} PDF documents")

    all_chunks = []

    for i, doc in enumerate(docs, start=1):
        doc_start = time.time()

        doc_chunks = list(
            chunk_text(
                text=doc["text"],
                meta={
                    "doc_id": doc["doc_id"],
                    "source": "rbi_pdf"
                }
            )
        )

        all_chunks.extend(doc_chunks)

        elapsed = time.time() - doc_start
        avg_time = (time.time() - start_time) / i
        remaining = avg_time * (total_docs - i)

        print(
            f"📘 [{i}/{total_docs}] {doc['doc_id']} → "
            f"{len(doc_chunks)} chunks | "
            f"{elapsed:.1f}s | "
            f"ETA: {remaining/60:.1f} min"
        )

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    total_time = time.time() - start_time

    print("\n✅ Chunking completed")
    print(f"📦 Total chunks: {len(all_chunks)}")
    print(f"⏱ Total time: {total_time/60:.2f} minutes")
    print(f"💾 Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
