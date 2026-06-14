# Recall@5 driver wired to embed_query_st and the scikit_docs_st collection
"""Recall@5 against the MiniLM-built `scikit_docs_st` collection.

Mirrors `scripts/recall_at_5.py` with two changes:
- Imports `embed_query_st` from `scripts.embed_with_st` in place of
  `embedder.embed_query`.
- Queries the `scikit_docs_st` collection directly via
  `store.get_collection("scikit_docs_st").query(...)`, converting
  cosine distance to similarity inline the way `store.query` does.
"""
import csv, time
from pathlib import Path
from src import store
from scripts.embed_with_st import embed_query_st

# Golden set path, per-type subset sizes, and the MiniLM collection name
GOLDEN = Path("data/golden_set.csv")
SUBSET_SIZE = {"factual": 5, "procedural": 3, "conceptual": 2, "comparative": 2}
COLLECTION = "scikit_docs_st"

# Hit-any: any returned id starts with any expected prefix
def hit_any(returned_ids: list[str], expected_prefixes: list[str]) -> bool:
    return any(r.startswith(p) for r in returned_ids for p in expected_prefixes if p)

# Queries the parallel collection and converts cosine distance to similarity
def query_st(qv: list[float], n_results: int = 5) -> list[tuple[str, float]]:
    """Return list of (doc_id, similarity_score) sorted by similarity desc."""
    result = store.get_collection(COLLECTION).query(
        query_embeddings=[qv],
        n_results=n_results,
        include=["documents", "distances"],
    )
    ids = result["ids"][0]
    distances = result["distances"][0]
    pairs = [(doc_id, 1.0 - distance) for doc_id, distance in zip(ids, distances)]
    pairs.sort(key=lambda p: p[1], reverse=True)
    return pairs

def main() -> None:
    # Pick the first SUBSET_SIZE[type] rows of each query_type
    rows = list(csv.DictReader(GOLDEN.open()))
    remaining = dict(SUBSET_SIZE)
    picked = []
    for row in rows:
        if remaining.get(row["query_type"], 0) > 0:
            picked.append(row)
            remaining[row["query_type"]] -= 1
    # Embed each question via MiniLM, query top-5, score hit-any, print
    start = time.monotonic()
    hits = 0
    hits1 = 0
    for i, row in enumerate(picked, 1):
        qv = embed_query_st(row["question"])
        returned = [doc_id for doc_id, _ in query_st(qv, n_results=5)]
        expected = [p for p in row["expected_doc_ids"].split("|") if p]
        is_hit = hit_any(returned, expected)
        hits += is_hit
        # Strict hit@1 on the first non-seeded result — mirrors recall_at_5.py
        # (the st collection carries no seeded chunks, so the filter is a no-op)
        top1 = next((r for r in returned if not r.startswith("seeded.")), returned[0])
        hits1 += hit_any([top1], expected)
        print(f"Q{i:>2} [{row['query_type']:<11}] {'HIT ' if is_hit else 'MISS'} {row['question'][:70]}")
    # Print aggregate recall@5, strict hit@1, plus runtime ($0 — local embedder)
    elapsed = time.monotonic() - start
    print(f"\nrecall@5: {hits}/{len(picked)} = {hits/len(picked):.2f}")
    print(f"hit@1 (excluding seeded.*): {hits1}/{len(picked)} = {hits1/len(picked):.2f}")
    print(f"runtime:  {elapsed:.1f}s ({len(picked)} queries, $0.00000 [local])")

if __name__ == "__main__":
    main()
