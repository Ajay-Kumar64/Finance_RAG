import time
import pickle
from rag.es_index import index_chunks  # your ES helper

OUTPUT_PATH = "artifacts/colab_chunks.pkl"
BATCH_SIZE = 50  # you can increase to 100+ if ES can handle

def main():
    with open(OUTPUT_PATH, "rb") as f:
        chunks = pickle.load(f)

    total_chunks = len(chunks)
    print(f"📦 Total chunks to index: {total_chunks}")
    start_time = time.time()

    for i in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_start = time.time()

        # Index the batcha
        index_chunks(batch)

        elapsed = time.time() - batch_start
        avg_time = (time.time() - start_time) / (i + len(batch))
        remaining = avg_time * (total_chunks - (i + len(batch)))

        print(
            f"📌 [{i + len(batch)}/{total_chunks}] Indexed batch of {len(batch)} chunks | "
            f"{elapsed:.2f}s | "
            f"ETA: {remaining/60:.2f} min"
        )

    total_time = time.time() - start_time
    print("\n✅ All chunks indexed")
    print(f"⏱ Total time: {total_time/60:.2f} minutes")

if __name__ == "__main__":
    main()
