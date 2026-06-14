# Solution notes — Module 26 (Latency Optimization)

This is the last implementation module. Every feature the prior modules
wired in is already in the starter: tracing (`src/tracing.py`), the
semantic cache (`src/cache/`), the FastAPI gateway with tier classifier
(`src/gateway/`), the cost dashboard (`src/cost/dashboard.py`), the
streaming endpoint (`src/streaming.py`), and the HNSW collection setup
(`src/store.py`). This module does not add new components; it **profiles**
them. The only new files these notes author are the two scratch
scripts the exercises name explicitly. The rest of the deliverables are
writeup tables and interpretive paragraphs.

## Files these notes add

| File | Maps to |
|---|---|
| `scripts/ttft_compare.py` | Exercise 2 — TTFT comparison client (blocking vs streaming) |
| `scripts/ef_search_sweep.py` | Exercise 3 — sandbox-collection `ef_search` sweep |

Both scripts are direct transcriptions of the fenced code blocks in
`INSTRUCTIONS.md`. The instructions tell learners to place them in
"scratch space — do not modify the starter tree" — that warning protects
the starter from drift while learners iterate. In these notes they live
in `scripts/` so they are discoverable next to the other operational
scripts (`run_eval.py`, `seed_cost_log.py`, etc.).

Run them after `make setup && make load-data` with `make serve` up in
another terminal:

```bash
uv run python -c "from src.cache import clear; clear()"   # cold blocking
uv run python scripts/ttft_compare.py
uv run python scripts/ef_search_sweep.py
```

## Writeup-only deliverables (no code to commit)

### Exercise 1 — Cold-versus-cached latency table

Expected shape from the Phoenix UI (or `make show-traces`) after the
five-query repeat against `/query`. Magnitudes vary by region, hosted
endpoint load, and Vocareum-vs-direct path; the **shape** is what the
rubric §7 evidence target wants:

```
| Span                    | Cold (ms)  | Cached (ms) |
|-------------------------|------------|-------------|
| classifier (LLM)        |   ~1300    |    n/a      |
| cache lookup            |    ~800    |    ~800     |
| retrieve (embed+search) |    ~900    |    n/a      |
| generator (LLM)         |   ~2200    |    n/a      |
| TOTAL                   |   ~5200    |    ~800     |
```

Speedup: `cold_total / cached_total ≈ 4-10×`, larger on a fast direct
endpoint and smaller on the Vocareum proxy. `route_query` runs the cache
lookup first, so a hit returns before the classifier or any other LLM
call fires; only the embedding round-trip inside the lookup is paid. The
cached total is therefore the embedding-lookup time (hundreds of ms on a
hosted endpoint), not near zero — hedge the ratio accordingly.

The one-paragraph interpretation should name:
- The generator span dominates the cold total (the LLM call is the
  wall-clock floor; everything else is single-digit to low-tens of ms).
- The cache hit eliminates the classifier, retrieval, and generator
  spans in one move — only the embedding lookup itself fires on the hit
  path, and on a hosted endpoint that lookup is the floor on cached latency.
- The cache wins are bounded by the workload's repeat rate; on
  paraphrase-heavy traffic (FAQ, docs Q&A) the hit rate climbs and the
  compression is real. On unique-query traffic (one-off research) the
  hit rate is near zero and the cache adds 25 ms of lookup overhead for
  no payoff. The semantic-caching module's exercises and the cost-monitoring
  module's cost report both anchor this hit-rate-versus-cost trade.

### Exercise 2 — TTFT-versus-total table

Expected shape from `uv run python scripts/ttft_compare.py` with the
cache cleared between calls (so blocking misses):

```
|               | TTFT (ms)  | Total (ms) |
|---------------|------------|------------|
| blocking      |   ~2500    |   ~2500    |
| streaming     |   ~500     |   ~2500    |
```

The right interpretation paragraph (two paragraphs total per the
exercise spec):

> Blocking TTFT equals blocking total — the urllib client cannot read
> any byte until the server flushes the whole response body. Streaming
> TTFT lands in the few-hundred-millisecond range because the first SSE
> `data:` frame ships as soon as the model emits its first token. Total
> time is comparable because the model still has to finish generating
> regardless of the response shape. The user-perceived difference is
> the spinner that did not appear, not the total clock. The engineering
> cost is the deferred output guards — the hallucination judge and the
> off-topic check that gate the blocking route fire on the complete
> answer and cannot run on a half-streamed token sequence without
> rewriting their contract. The starter offers both routes so the
> trade-off is visible in code.

> Default the docs-FAQ workload to `/query` (blocking). The output
> guards are load-bearing for a docs assistant where hallucinated API
> signatures cost the user real debugging time, and the few seconds
> of spinner sits within the acceptable range for a documentation tool.
> Reach for `/query/stream` on the surfaces where perceived-latency
> matters more than guardrail enforcement — a tutoring tutor-style
> conversation, an exploratory assistant where the user is sketching
> ideas — and accept that the output guards are deferred or omitted
> on that route. Most teams pick one as the default and reach for the
> other when product surface area justifies the operational cost of
> running both.

### Exercise 3 — `ef_search` sweep table

Expected shape from `uv run python scripts/ef_search_sweep.py` plus the
hand-built recall@5 check. Mean latencies are sensitive to host CPU,
disk contention, and whether the embeddings are warm in the OS page
cache; the **ordering** (`ef=200` slower than `ef=10`) is what should
be reproducible:

```
| ef_search | mean latency (ms) | recall@5 |
|-----------|-------------------|----------|
|    10     |       ~5.7        |   1.0    |
|    50     |       ~6.1        |   1.0    |
|   200     |       ~6.4        |   1.0    |
```

The right interpretation paragraph:

> At ~750 chunks the curve barely moves — `ef=200` is only a few percent
> slower than `ef=10` on the per-query mean, a fraction of a millisecond,
> and the absolute numbers are single-digit milliseconds across the sweep.
> On a noisy run the gaps can even reorder; only the broad ordering holds. The
> generator span from Exercise 1 dominates the wall-clock budget by
> two orders of magnitude, so tuning effort spent here saves single
> percentage points off total latency at best. Recall@5 saturates by
> `ef=50` on this corpus, which is the typical shape on the Sift1M
> benchmark Pinecone publishes — the curve flattens once `ef_search`
> exceeds the local-graph degree by a comfortable multiple. The
> operational read is that cache hit rate (the semantic-caching module)
> and model choice (the cost-monitoring module's cost report) are the
> two levers worth tuning first; vector search is the third lever at
> three-thousand-chunk scale, and only climbs the priority list as the
> corpus grows past a few million rows.

> On a hypothetical ten-million-row workload, the first knob to tune is
> still the cache, then the model choice — both compound multiplicatively
> against per-request cost. The HNSW knobs become operationally meaningful
> when the index size makes the search latency comparable to the embed
> call, which on Chroma is usually past the ten-million-row mark and
> depends on the dimension of the embedding and the host's RAM-versus-disk
> ratio. Pinecone's HNSW article walks the recall-versus-latency curve
> on the Sift1M benchmark; that is the shape you would replicate against
> your own workload before committing to an `ef_search` value in
> production.

## Verification

```bash
# After make setup + make load-data + make serve running:
uv run python -c "from src.cache import clear; clear()"
uv run python scripts/ttft_compare.py
# Compare TTFT vs total for both rows; streaming TTFT < blocking TTFT.

# Independently:
uv run python scripts/ef_search_sweep.py
# Expect three rows with ef_search ordering 10 < 50 < 200 on mean_ms.

# Cleanup sandbox collections so the next exercise's vector DB is unchanged:
uv run python -c "
import chromadb
from chromadb.config import Settings as CS
from src.config import settings
c = chromadb.PersistentClient(path=settings.chroma_path, settings=CS(anonymized_telemetry=False))
for ef in [10, 50, 200]:
    try: c.delete_collection(f'scikit_docs_ef_{ef}')
    except Exception: pass
print('sandbox collections removed')
"
```

## KNOWN-LIMITATIONs

None. The two authored scripts are direct transcriptions of the fenced
code blocks in `INSTRUCTIONS.md`. The Exercise 1, 2, and 3 writeups are
inherently observational — every learner produces different magnitudes
on a different hosted endpoint at a different time of day — so the
"expected shape" tables above are illustrative; the graded artifact is
the learner's own table plus the interpretive paragraphs that read it.

Note on the Exercise 3 recall metric: the `GOLDEN` list in the
instructions uses `doc_id` strings of the form `ensemble/forest.rst#L42`
that are illustrative — the actual `doc_id` values depend on the
ingest run's chunking output. Learners are expected to pick five chunks
they have inspected from a live `make show-traces` output rather than
copy-paste the placeholder ids. This is called out in the instructions
("pick chunks you have inspected from the live collection") and is the
correct shape for the recall metric, not a gap in these notes.
