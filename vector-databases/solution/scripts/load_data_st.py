"""MiniLM rebuild into a parallel collection for Exercise 3."""
import time
from pathlib import Path
from src import store
from src.chunker import chunk_doc
from src.corpus import load_corpus
from scripts.embed_with_st import _model

COLLECTION = "scikit_docs_st"

def main():
    start = time.monotonic()
    chunks = [c for sec in load_corpus(Path("data/scikit-learn-cache"), "manual")
              for c in chunk_doc(sec)]
    texts = [c["text"] for c in chunks]
    embeddings = _model.encode(
        texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True
    ).tolist()
    col = store.get_collection(COLLECTION)
    col.upsert(
        ids=[c["chunk_id"] for c in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"doc_id": c["chunk_id"]} for c in chunks],
    )
    print(f"upserted {len(chunks)} into '{COLLECTION}' in {time.monotonic()-start:.1f}s")

if __name__ == "__main__":
    main()
