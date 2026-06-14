# Recall@5 over a balanced 12-question subset of the golden set
"""Recall@5 against a balanced subset of the golden set."""
import csv, time
from pathlib import Path
from src import embedder, store

# Golden set path and per-type subset bucket sizes
GOLDEN = Path("data/golden_set.csv")
SUBSET_SIZE = {"factual": 5, "procedural": 3, "conceptual": 2, "comparative": 2}

# Hit-any: any top-5 doc_id starts with any expected prefix
def hit_any(returned_ids: list[str], expected_prefixes: list[str]) -> bool:
    return any(r.startswith(p) for r in returned_ids for p in expected_prefixes if p)

def main() -> None:
    # Pick the first SUBSET_SIZE[type] rows of each query_type
    rows = list(csv.DictReader(GOLDEN.open()))
    remaining = dict(SUBSET_SIZE)
    picked = []
    for row in rows:
        if remaining.get(row["query_type"], 0) > 0:
            picked.append(row)
            remaining[row["query_type"]] -= 1
    # Embed each question, query top-5, score hit-any, print per-question line
    start = time.monotonic()
    hits = 0
    hits1 = 0
    for i, row in enumerate(picked, 1):
        qv = embedder.embed_query(row["question"])
        returned = [s.doc_id for s in store.query(qv, n_results=5)]
        expected = [p for p in row["expected_doc_ids"].split("|") if p]
        is_hit = hit_any(returned, expected)
        hits += is_hit
        # Strict hit@1 on the first non-seeded result — measures the real
        # corpus whether or not `make seed-difficulty` has run
        top1 = next((r for r in returned if not r.startswith("seeded.")), returned[0])
        hits1 += hit_any([top1], expected)
        print(f"Q{i:>2} [{row['query_type']:<11}] {'HIT ' if is_hit else 'MISS'} {row['question'][:70]}")
    # Print aggregate recall@5, strict hit@1, and the runtime/cost summary
    elapsed = time.monotonic() - start
    print(f"\nrecall@5: {hits}/{len(picked)} = {hits/len(picked):.2f}")
    print(f"hit@1 (excluding seeded.*): {hits1}/{len(picked)} = {hits1/len(picked):.2f}")
    print(f"runtime:  {elapsed:.1f}s ({len(picked)} queries, ${len(picked)*2e-5:.5f})")

if __name__ == "__main__":
    main()
