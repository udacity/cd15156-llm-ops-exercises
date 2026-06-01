# Rebuilds the corpus into a parallel scikit_docs_st collection for comparison
"""MiniLM rebuild into a parallel collection for Exercise 3."""
import time
from pathlib import Path
from src import store
from src.chunker import chunk_doc
from src.corpus import load_corpus
from scripts.embed_with_st import _model

# Distinct collection name keeps the OpenAI collection intact
COLLECTION = "scikit_docs_st"

def main():
    # Chunk the cached corpus with the production chunker
    start = time.monotonic()
    chunks = [c for sec in load_corpus(Path("data/scikit-learn-cache"), "manual")
              for c in chunk_doc(sec)]
    # Batch-encode with MiniLM, normalizing for cosine similarity
    texts = [c["text"] for c in chunks]
    embeddings = _model.encode(
        texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True
    ).tolist()
    # Upsert ids, documents, embeddings, and metadatas into the parallel collection
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
