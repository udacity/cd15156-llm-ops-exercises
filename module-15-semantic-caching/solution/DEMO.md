> A walkthrough of the codebase you'll work with. See INSTRUCTIONS.md for the exercise tasks.

# Module 15 — Demo: Wire the Chroma Cache, Watch a Paraphrase Hit, Tune the Threshold

Module 14 named the architecture in four moves — embed the query, similarity-search the cache, return on hit, call the model and write back on miss — and named the one parameter that decides whether the whole thing earns its keep: the similarity threshold. This demo turns that into running code in the ScikitDocs starter. You will look at the cache module, fire a query that warms the cache, fire a paraphrase that hits it, then walk the threshold up and down and watch the hit-miss boundary move. The pipeline does not have an HTTP route yet — the gateway module is where that lands — so this module composes the cache around `run_pipeline` directly in Python. The composition is identical to what the gateway will reproduce inline at the HTTP boundary; doing it in Python first makes the layering legible.

## Setup

With `make setup` complete and `make load-data` reporting the scikit-learn docs are in Chroma at `data/chroma`, you should be able to import `run_pipeline` and answer a question end-to-end. Your `.env` carries the same two values every prior module needed:

```
OPENAI_API_KEY=voc-...           # or sk-... on a direct OpenAI account
OPENAI_BASE_URL=https://openai.vocareum.com/v1   # omit if you are not on Vocareum
```

`OPENAI_BASE_URL` is read by `settings.openai_base_url` at `src/config.py:30` and threaded into the shared OpenAI client used by both the chat completion in `src/generator.py` and the embedder in `src/embedder.py`. The same `text-embedding-3-small` model embeds the section chunks during ingestion and the cache queries at runtime — same vector space, comparable distances. That matters: the cache's similarity scores only mean what they appear to mean if the embedder on the write path and the embedder on the read path are the same model. The starter enforces that by routing both calls through one shared `embed_query` function at `src/embedder.py:69`.

One detail to flag before you fire anything. The cache writes happen *after* the LLM call returns, and reads bypass the LLM entirely on hits. That ordering is what the cost log will reflect: a cache miss appends one row to `data/cost_log.jsonl` (the LLM call we did make); a cache hit appends nothing (no model call to bill). Module 13's cost stack and Module 15's cache stack interact at exactly this seam, and Exercise 3 of this module turns the seam into a dollars-per-query measurement.

## Walkthrough 1 — Read the cache module top to bottom

Open `src/cache/semantic.py`. It is roughly 140 lines and it is the whole cache. The shape mirrors the four-move architecture from Module 14 cleanly.

Lines 1-32 are the module docstring and the `COLLECTION_NAME = "cache"` constant. That literal is the rubric §10 evidence handle — when you list cache contents via `get_or_create_collection("cache")`, the rubric grader expects exactly that name. The `_client()` helper at lines 56-68 is a `lru_cache`-singleton `chromadb.PersistentClient` pointed at the same `settings.chroma_path` directory as the document corpus. One on-disk Chroma store, two collections — `scikit_docs` for retrieval and `cache` for cached query-response pairs. The `_collection()` helper at lines 71-76 opens the cache collection with `metadata={"hnsw:space": "cosine"}` so the distance Chroma returns is cosine, and `1.0 - distance` is the similarity in `[0, 1]` that the threshold lives in. Module 14 named this convention; the file pins it.

Lines 79-121 are `lookup(question, *, threshold=constants.CACHE_SIMILARITY_THRESHOLD)`. Read it line by line:

```python
def lookup(question, *, threshold=constants.CACHE_SIMILARITY_THRESHOLD):
    collection = _collection()
    if collection.count() == 0:
        return None

    embedding = embed_query(question)
    results = collection.query(query_embeddings=[embedding], n_results=1, ...)
    ids = results["ids"][0]
    if not ids:
        return None

    distance = float(results["distances"][0][0])
    similarity = 1.0 - distance
    if similarity < threshold:
        return None
    ...
```

The cold-start short-circuit at line 96 — `if collection.count() == 0` — is the small detail that catches most learners later. An empty cache skips the embedding call entirely; on a fresh deployment, the first request does not pay the embed cost for a lookup that cannot hit. Then the embed-and-query at lines 99-103 is the architectural primitive Module 14 named. `n_results=1` is the right choice here because the threshold decision is on the single nearest neighbor; we are not ranking, we are gating. The distance-to-similarity conversion at lines 109-110 and the threshold gate at line 111 are the cache's hit-or-miss logic in three lines.

Lines 113-119 are the lazy TTL eviction. Module 14's invalidation video named the pattern: stamp each entry with a `ttl_s` and a `created_at`, and on read, if the nearest match is older than its TTL, delete it inline and report a miss. No background sweep, no surprise eviction. An entry stamped with `ttl_s == 0` is treated as immortal — useful for stable answers you trust to age well.

Lines 121-122 are the hit-return path. The cached `QueryResponse` is deserialised from the metadata blob and tagged with `cached=True` before returning. That flag is the rubric §10 evidence signal — `cached` is on the response schema so the grader can see the cache engaged on each paraphrase. The `store` function at lines 125-160 is the same primitive on the write side: embed, upsert with a UUID key, four metadata fields (`question`, `response_json`, `created_at`, `ttl_s`), return the key. The defaults — one-hour TTL — are a reasonable starting point for a docs-FAQ workload; the exercises ask you to think about when you would set them differently. `clear()` at lines 163-172 drops every entry and returns the count removed — used by the exercises to reset between runs.

The `cached_route_query` helper at `src/cache/wrapper.py:25-50` composes the cache around `run_pipeline` so Module 15's demo and exercises have a clean entry point without waiting for the gateway module. The gateway module will reproduce the same composition inline at the request boundary — input guards, then cache lookup on the cleaned text, then the tiered router on miss, then output guards, then the conditional cache store only after guards pass. The safe-default that comes out of that composition — never write a response from an input that tripped a guard — is what the cache-poisoning discussion in Module 14 anchored.

## Walkthrough 2 — Warm the cache, fire a paraphrase, watch the hit

From a Python shell with the cache cleared, fire one warmup query:

```
uv run python -c "
from src.cache import clear, cached_route_query
print('removed:', clear())
r = cached_route_query('What is the default criterion for RandomForestRegressor?')
print('cached:', r.cached, '/ answer head:', r.answer[:80])
"
```

The response field `cached: False` is the miss path: route through the LLM, get an answer, write to the cache. The answer head should mention `squared_error` (the default since scikit-learn 1.0; `"mse"` was the pre-1.0 name). The cost log just gained one row — verify with `wc -l data/cost_log.jsonl`.

Now fire a paraphrase:

```
uv run python -c "
from src.cache import cached_route_query
r = cached_route_query('Default split criterion in RandomForestRegressor?')
print('cached:', r.cached, '/ answer head:', r.answer[:80])
"
```

The response field `cached: True` is your hit indicator. The `answer` is the same answer the first query produced — same text, same sources, same cost reported (the cached row carries the original call's cost so the response shape stays consistent). Tail the cost log again with `wc -l data/cost_log.jsonl`; the line count did not grow. No LLM call was made, nothing was logged. That is the cache earning its keep — the call you didn't make is the call you didn't pay for.

Confirm the cache contents directly:

```
uv run python -c "
from chromadb import PersistentClient
from chromadb.config import Settings as CS
c = PersistentClient(path='data/chroma', settings=CS(anonymized_telemetry=False))
col = c.get_or_create_collection('cache')
print('count:', col.count())
for m in col.get(limit=10)['metadatas']:
    print(' ', m['question'][:60])
"
```

You should see one entry, the original warmup question. The paraphrase hit that one entry; it did not create a second cache row. That is the canonical cache-hit pattern — one cached answer serves many paraphrases of the same intent.

## Walkthrough 3 — Walk the threshold up and down

The threshold lives on `lookup`'s keyword argument at `src/cache/semantic.py:79`. The default is `constants.CACHE_SIMILARITY_THRESHOLD` (0.85). To demonstrate threshold sensitivity, call `lookup` directly from a small Python shell against the cache you just warmed in Walkthrough 2.

Run a threshold sweep against three test queries — a tight paraphrase, a terse paraphrase, and a near-miss against a different sklearn estimator:

```
uv run python -c "
from src.cache.semantic import lookup
queries = [
    'Default split criterion in RandomForestRegressor?',
    'RandomForestRegressor default criterion?',
    'What is the default criterion for RandomForestClassifier?',
]
for q in queries:
    print(f'\nquery: {q}')
    for t in [0.70, 0.85, 0.95]:
        hit = lookup(q, threshold=t)
        print(f'  threshold={t}: {\"HIT\" if hit else \"MISS\"}')
"
```

What you should see, and why each row lands the way it does. The first query — a clean paraphrase of the warmup — hits at all three thresholds. The cosine similarity sits in the high 0.90s because the embedding model maps `"what is the default criterion for X"` and `"default split criterion in X"` into near-identical vector neighborhoods. This is the case the cache is built for.

The second query — terser phrasing — typically hits at 0.85 and 0.70 and may miss at 0.95. Word-order shifts and dropped function words pull the cosine down a few hundredths into the 0.88-0.93 range, depending on the embedder run. At 0.95 the cache becomes a paraphrase pedant; at 0.85 it forgives.

The third query — about a different scikit-learn estimator — should miss at 0.95 and at 0.85, and may hit at 0.70. That last hit is the teaching moment. The cached answer is about the regressor's default criterion (`squared_error`), the query is about the classifier's default criterion (`gini`), and at 0.70 the cache will happily return regressor-flavored content to a classifier question because both sentences share the structural shape `"what is the default criterion for RandomForest<X>"` and the embedding model puts them within 0.30 cosine distance. That is the wrong-answer mode Module 14 named. Threshold is a quality knob first and a cost knob second; a 0.70 setting is where the savings stop being savings.

The default sits at 0.85 because it is the typical baseline for sentence-transformer-class embeddings on FAQ-style workloads. Portkey's managed gateway defaults higher, around 0.95, which biases the opposite direction — higher precision, lower hit rate. There is no universal right number. Exercise 1 runs a fifteen-paraphrase sweep on the default and reports the hit rate; Exercise 2 builds a deliberate near-miss out of the RandomForest classifier-vs-regressor pair to feel the wrong-answer mode in your own hands; Exercise 3 pairs the cache hits with Module 13's cost log to compute the dollars-per-query delta.

---

