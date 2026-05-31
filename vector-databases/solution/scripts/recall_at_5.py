# TODO(m05-ex2): write scripts/recall_at_5.py — recall@5 over a balanced
# 12-question golden subset (5 factual / 3 procedural / 2 conceptual / 2 comparative).
"""Recall@5 against a balanced subset of the golden set."""
import csv, time
from pathlib import Path
from src import embedder, store

# TODO(m05-ex2): point GOLDEN at data/golden_set.csv and pick the subset bucket sizes.
GOLDEN = Path("data/golden_set.csv")
SUBSET_SIZE = {"factual": 5, "procedural": 3, "conceptual": 2, "comparative": 2}

# TODO(m05-ex2): implement hit-any semantics — any top-5 doc_id starts with any expected prefix.
def hit_any(returned_ids: list[str], expected_prefixes: list[str]) -> bool:
    return any(r.startswith(p) for r in returned_ids for p in expected_prefixes if p)

def main() -> None:
    # TODO(m05-ex2): pick the first SUBSET_SIZE[type] rows of each query_type from the golden CSV.
    rows = list(csv.DictReader(GOLDEN.open()))
    remaining = dict(SUBSET_SIZE)
    picked = []
    for row in rows:
        if remaining.get(row["query_type"], 0) > 0:
            picked.append(row)
            remaining[row["query_type"]] -= 1
    # TODO(m05-ex2): embed each question, query the store top-5, score hit-any, print per-question line.
    start = time.monotonic()
    hits = 0
    for i, row in enumerate(picked, 1):
        qv = embedder.embed_query(row["question"])
        returned = [s.doc_id for s in store.query(qv, n_results=5)]
        expected = [p for p in row["expected_doc_ids"].split("|") if p]
        is_hit = hit_any(returned, expected)
        hits += is_hit
        print(f"Q{i:>2} [{row['query_type']:<11}] {'HIT ' if is_hit else 'MISS'} {row['question'][:70]}")
    # TODO(m05-ex2): print the aggregate recall@5 and runtime/cost summary.
    elapsed = time.monotonic() - start
    print(f"\nrecall@5: {hits}/{len(picked)} = {hits/len(picked):.2f}")
    print(f"runtime:  {elapsed:.1f}s ({len(picked)} queries, ${len(picked)*2e-5:.5f})")

if __name__ == "__main__":
    main()
