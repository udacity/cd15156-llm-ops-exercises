# Embedder comparison (Exercise 3)

Twelve-question golden subset. OpenAI's `scikit_docs` collection versus the MiniLM
`scikit_docs_st` rebuild.

| Embedder | dim | recall@5 | strict hit@1 | corpus-build cost | query latency (median) |
|---|---|---|---|---|---|
| `text-embedding-3-small` | 1536 | 12/12 = 1.00 | 9/12 = 0.75 | ~$0.10 (cold) / $0 (warm cache) | ~1.5s (network) |
| `all-MiniLM-L6-v2` | 384 | 12/12 = 1.00 | 6/12 = 0.50 | $0 (local) | ~0.05s (in-process) |

Why the rebuild is unavoidable: a Chroma collection fixes its vector dimension on the first
insert, so the 1536-dim `scikit_docs` collection rejects a 384-dim MiniLM query with
`InvalidDimensionException`. Swapping embedders therefore means building a parallel collection at
the new dimension (`scikit_docs_st`) and pointing reads at it, never re-embedding in place.
