"""Exercise 3 — sweep `ef_search` against parallel sandbox collections.

Pulls every chunk out of the live `scikit_docs` collection, copies them into
three parallel collections built with `ef_search` of 10, 50, and 200, then
runs the same five queries twenty times against each collection and reports
mean per-query latency. Embedding calls are done up front and excluded from
the timing so the measurement is the vector search alone.

Does not touch `src/store.py` or the live `scikit_docs` collection. The
parallel collections are dropped between iterations and can be cleaned up
afterward with the snippet at the bottom of Exercise 3.

Run with `make load-data` already complete:

    uv run python scripts/ef_search_sweep.py
"""

# Build sandbox collections at ef_search of 10/50/200, replay the same queries
# against each, and print mean per-query latency.
import sys
import time
from pathlib import Path

import chromadb

# chromadb 0.6.x calls posthog.capture() positionally, but posthog>=6 made
# those args keyword-only, so every telemetry send raises and chromadb logs
# "Failed to send telemetry event ...". anonymized_telemetry=False does not
# suppress it (verified), so quiet the telemetry logger directly.
import logging
import posthog
posthog.disabled = True  # hard-off: never construct/send chromadb product telemetry
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
from chromadb.config import Settings as CS

# Make the project root importable when run directly (e.g.
# `uv run python scripts/ef_search_sweep.py`). The make targets get this from
# the Makefile's `export PYTHONPATH := .`; a direct script invocation does not.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.embedder import embed_query

client = chromadb.PersistentClient(
    path=settings.chroma_path,
    settings=CS(anonymized_telemetry=False),
)

src = client.get_or_create_collection("scikit_docs").get(
    include=["documents", "embeddings", "metadatas"]
)

QUERIES = [
    "How do I tune n_estimators in RandomForestClassifier?",
    "What is the default criterion for DecisionTreeRegressor?",
    "How do I use ColumnTransformer with Pipeline?",
    "What does the random_state parameter do?",
    "How do I serialize a fitted estimator?",
] * 20  # 100 query repetitions


def sweep(ef: int) -> dict:
    name = f"scikit_docs_ef_{ef}"
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.get_or_create_collection(
        name,
        metadata={"hnsw:space": "cosine", "hnsw:search_ef": ef},
    )
    col.upsert(
        ids=src["ids"],
        documents=src["documents"],
        embeddings=src["embeddings"],
        metadatas=src["metadatas"],
    )
    embeddings = [embed_query(q) for q in QUERIES]
    start = time.perf_counter()
    for emb in embeddings:
        col.query(query_embeddings=[emb], n_results=5)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {"ef_search": ef, "mean_ms": elapsed_ms / len(QUERIES)}


if __name__ == "__main__":
    for ef in [10, 50, 200]:
        print(sweep(ef))
