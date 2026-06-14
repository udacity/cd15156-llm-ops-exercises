# Module 05 — Solution Notes

These notes add four scripts to the ScikitDocs starter under `scripts/`:

- `scripts/recall_at_5.py` — Exercise 2's twelve-question recall@5 measurement against the OpenAI-built `scikit_docs` collection.
- `scripts/embed_with_st.py` — Exercise 3 step 1's sentence-transformers wrapper exposing `embed_query_st` over the local `all-MiniLM-L6-v2` model.
- `scripts/load_data_st.py` — Exercise 3 step 4's rebuild into a parallel `scikit_docs_st` collection using MiniLM (384-dim) so the OpenAI collection stays intact for comparison.
- `scripts/recall_at_5_st.py` — Exercise 3 step 5's MiniLM-side recall@5, mirroring `recall_at_5.py` with two changes: imports `embed_query_st` and queries the `scikit_docs_st` collection directly with inline cosine-distance → similarity conversion.

## Exercise 1 — expected output

After `rm -rf data/chroma && make load-data`:

- `make load-data` prints four timing lines (clone, parse, embed, upsert) and a final `done — N chunks upserted, total Ms` line. On a warm embedding cache N is ~747, the total is under ten seconds.
- `store.get_collection().count()` returns the same N — ~747 without `make seed-difficulty`, ~755 if seed-difficulty ran previously.
- `peek(limit=1)` shows a 1536-dimensional embedding and metadata keys including `source_path`, `section_title`, `section_path`, `url`, `has_code`, `code_languages`, `xrefs`, `scikit_learn_sha`.

## Exercise 2 — expected output

`PYTHONPATH=. uv run python scripts/recall_at_5.py` prints a 12-row HIT/MISS table, `recall@5: 12/12 = 1.00`, and `hit@1 (excluding seeded.*): 9/12 = 0.75` in roughly eighteen seconds. Recall@5 is not the binding constraint on retrieval quality at this corpus size — the hit@1 line is where quality differences show; the seeded-chunk top-1 displacement is what a later evaluation module will measure.

## Exercise 3 — expected output and cost-vs-quality recommendation

The cold MiniLM rebuild via `PYTHONPATH=. uv run python scripts/load_data_st.py` takes ~90 seconds on CPU and populates `scikit_docs_st` with ~747 chunks at 384 dimensions. `PYTHONPATH=. uv run python scripts/recall_at_5_st.py` then reports recall@5 12/12 = 1.00 — identical to OpenAI — and strict hit@1 around 6/12 = 0.50 versus OpenAI's 9/12 = 0.75 on the same twelve-question subset.

Cost-vs-quality recommendation for the ScikitDocs corpus: **stay on `text-embedding-3-small`**. Recall@5 ties at 1.00 on this corpus; the quality gap shows at strict top-1 (9/12 vs 6/12) — the rank the generator consumes first — and the cold-build cost (~$0.10) is one-time because the embedding cache makes warm rebuilds free. The MiniLM path is the right call when (a) the corpus is large enough that per-build cost is no longer rounding error, (b) regulatory or air-gap constraints forbid sending the corpus to a hosted API, or (c) sub-50ms query latency is a hard product requirement. None of those apply to ScikitDocs. The two-collection blue/green setup remains useful regardless — a later RAGOps module reuses it for embedder upgrades (e.g., `text-embedding-3-small` → `text-embedding-3-large`), which suffer the same dimensional-mismatch rebuild problem as the OpenAI → MiniLM swap.

## KNOWN-LIMITATIONs

- **`scripts/recall_at_5_st.py`** was derived from the prose in Exercise 3 step 5 ("copy `recall_at_5.py` ... change two things") rather than from a complete fenced code block. The interpretation here imports `embed_query_st` directly and inlines the distance → similarity conversion in a small `query_st` helper because the public `store.query` is hard-pinned to the default `scikit_docs` collection. Alternative implementations are valid; the acceptance criterion (a side-by-side recall table) is what matters.
- **Reference numbers were re-measured live (REQ-103, warm cache).** recall@5 = 12/12 for both embedders; strict hit@1 = 9/12 (OpenAI) vs 6/12 (MiniLM). The earlier spec figure of 10/12 = 0.83 MiniLM recall did not reproduce — recall@5 saturates on this 747-chunk corpus, so the comparison was re-anchored at top-1. The ~90s cold MiniLM rebuild figure is a cold-cache estimate; warm-cache rebuilds here run ~13s.
