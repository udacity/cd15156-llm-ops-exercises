# Module 15 — Implement Semantic Caching with Chroma

## Setup (read first)

This starter is the ScikitDocs RAG app with the prompt loader, vector DB, RAG pipeline, in-process Phoenix tracing, RAGAS evaluation harness, cost monitoring, semantic caching (`src/cache/`), FastAPI gateway, guardrails, A/B testing, RAGOps watchers, and latency optimizations already wired. In this module you will: (1) run a fifteen-query paraphrase set through `cached_route_query` and report the hit rate, (2) construct a deliberate near-miss query and sweep the threshold at 0.70 / 0.85 / 0.95 to surface the wrong-answer mode, and (3) pair the cache hits with Module 13's cost log to compute a dollars-per-query delta plus a monthly projection. There is no new code to author in this module — the cache primitives (`lookup`, `store`, `clear`, `cached_route_query`) ship complete in `src/cache/`; your deliverables are three writeup artifacts. Run `make setup`, then `make load-data` to bring up the corpus, then follow the demo walkthrough and the exercise tasks below.

---

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

# Module 15 — Exercise: Hit Rate, Near-Miss, Cost Delta

The demo wired the cache into your hands at one query at a time. These exercises run real paraphrase sets through `cached_route_query` and turn the cache's behavior into three artifacts: a hit-rate report on a fifteen-query paraphrase set, a near-miss case that demonstrates why the threshold matters, and a cost delta that pairs the cache with Module 13's cost log. Plan for twenty minutes, weighted slightly toward Exercise 2 because the near-miss is the muscle memory worth burning in — a careless threshold is the failure mode that does not show up on a cost dashboard.

## Setup

Same as the demo. With `make setup` complete and `make load-data` reporting the corpus is in Chroma, your `.env` carrying `OPENAI_API_KEY` plus `OPENAI_BASE_URL` when you are on Vocareum. All three exercises drive the cache through `cached_route_query` (Python-direct, no gateway yet) and need a working OpenAI key. Budget a few cents on Vocareum or any other endpoint; on `gpt-4o` the fifteen-query Exercise 1 run costs well under five cents end-to-end, and Exercise 2 adds maybe one more cent. Exercise 3 is analysis on the data you already have.

Before you start, clear the cache from any prior runs so each exercise starts on a known state:

```
uv run python -c "from src.cache import clear; print('removed:', clear())"
```

The first run prints whatever count was sitting in the cache; subsequent runs print zero. Then keep a second terminal handy for inspecting `data/chroma` and `data/cost_log.jsonl` between runs.

## Exercise 1 — Fifteen paraphrases, hit-rate report

The starter's default cache threshold is 0.85, set on `src/cache/semantic.py:79` via `constants.CACHE_SIMILARITY_THRESHOLD`. The rubric §10 evidence target is six paraphrases with at least three hits. This exercise goes further — fifteen queries built as five base questions with three paraphrases apiece — so the hit-rate signal is statistically meaningful instead of binary. The expected pattern is five misses and ten hits: the first time each base question reaches the cache it misses (the cache is empty for that intent and the LLM call happens) and the two paraphrases that follow hit the prior write.

### What to do

1. Build the paraphrase set. Five base questions about distinct scikit-learn APIs, each with two paraphrased variants. The shape is the original question, paraphrase A, paraphrase B — three queries per base, fifteen total. Use this set (or substitute your own if you want different APIs, just keep the three-per-base structure):

   ```
   queries=(
     "What is the default criterion for RandomForestRegressor?"
     "Default split criterion in RandomForestRegressor?"
     "RandomForestRegressor default criterion?"

     "How many trees does RandomForestClassifier use by default?"
     "Default n_estimators for RandomForestClassifier?"
     "Number of trees in RandomForestClassifier default?"

     "What kernel does SVC use by default?"
     "Default kernel for SVC?"
     "SVC's default kernel choice?"

     "What scoring metric does cross_val_score use by default?"
     "Default scoring for cross_val_score?"
     "cross_val_score default scoring metric?"

     "What is the default test_size for train_test_split?"
     "Default test_size for train_test_split?"
     "train_test_split default test size?"
   )
   ```

   Order matters here. The first query in each group of three is the "original" the cache will miss on and write; the next two are the paraphrases that should hit that write. Running them in a different order changes which query lands in the cache as the original — the hit-rate total stays the same, but the per-query trace shifts.

2. Fire all fifteen queries through `cached_route_query` and capture each response's `cached` flag. From a Python shell:

   ```
   uv run python -c "
   from src.cache import cached_route_query
   queries = [
       'What is the default criterion for RandomForestRegressor?',
       'Default split criterion in RandomForestRegressor?',
       'RandomForestRegressor default criterion?',
       'How many trees does RandomForestClassifier use by default?',
       'Default n_estimators for RandomForestClassifier?',
       'Number of trees in RandomForestClassifier default?',
       'What kernel does SVC use by default?',
       'Default kernel for SVC?',
       \"SVC's default kernel choice?\",
       'What scoring metric does cross_val_score use by default?',
       'Default scoring for cross_val_score?',
       'cross_val_score default scoring metric?',
       'What is the default test_size for train_test_split?',
       'Default test_size for train_test_split?',
       'train_test_split default test size?',
   ]
   results = []
   for q in queries:
       r = cached_route_query(q)
       results.append((r.cached, q))
       print(f'{r.cached} | {q}')
   hits = sum(1 for c, _ in results if c)
   print(f'hits={hits}/{len(results)} ({100*hits/len(results):.1f}%)')
   "
   ```

   Each line prints `True | <question>` for a cache hit or `False | <question>` for a miss. Expect five `False` rows (the first of each group of three) and ten `True` rows.

3. Expected output: `hits=10/15 (66.7%)`. Some workloads land at 9/15 or 11/15 instead because one paraphrase may be too aggressive (a 0.84 cosine slips below 0.85) or because the embedder ranks unexpectedly close on a different base question's cached entry. Hit rates in the 60-to-75 percent range on a paraphrase-heavy set are within the workload-dependent envelope Regmi and Pun 2024 reported on customer-service traffic.

4. Confirm via the cache contents directly, the rubric §10 evidence pattern:

   ```
   uv run python -c "
   from chromadb import PersistentClient
   from chromadb.config import Settings as CS
   c = PersistentClient(path='data/chroma', settings=CS(anonymized_telemetry=False))
   col = c.get_or_create_collection('cache')
   print('cache entries:', col.count())
   for m in col.get(limit=10)['metadatas']:
       print('  -', m['question'])
   "
   ```

   You should see five entries — one per base question, the original first-arrival of each group — and the count matches the number of cache misses. The two paraphrases per group hit the one original write; they did not produce new cache rows.

### Acceptance criterion

Three artifacts in your writeup. First, the fifteen-row hit-or-miss table (one line per query with the `True/False` and the question text). Second, the hit-rate summary (`hits/total` plus the percentage). Third, the cache-contents dump showing five entries with the five base questions as the cached originals. A one-paragraph note interpreting the pattern: which paraphrases hit which originals, and whether any paraphrase missed when you expected a hit (the most common reason is a near-miss at 0.83 cosine against the 0.85 threshold; that case is what Exercise 2 deliberately constructs).

## Exercise 2 — The near-miss case

Exercise 1 stayed within the cache's correct-hit territory. This exercise constructs a deliberate near-miss — a query that is semantically close enough to a cached entry to score around 0.83 cosine but is asking about a different scikit-learn estimator with a different correct answer. At the default 0.85 threshold the cache misses (correctly). At a lowered 0.70 threshold the cache hits (incorrectly) and you get the wrong-answer mode Module 14 named. The exercise is to walk both sides of that boundary, look at what the cache would have served at the loose threshold, and put words to why that is the failure mode the threshold is guarding against.

### What to do

1. Leave the cache populated from Exercise 1. The five entries you wrote there are the substrate for the near-miss. You want the cache to have answers about specific scikit-learn APIs so the near-miss can land against a real cached answer that is wrong for the new question.

2. Construct the near-miss query. The shape that works reliably is a question that shares the structural template of a cached entry but swaps the estimator class. Use:

   ```
   near_miss="What is the default criterion for RandomForestClassifier?"
   ```

   That query is structurally near-identical to the cached "What is the default criterion for RandomForestRegressor?" from Exercise 1 — same noun, same verb, same question shape — but it is about the *classifier*, not the *regressor*. The cached answer talks about `squared_error` (the regressor's default split criterion since scikit-learn 1.0; `"mse"` was the pre-1.0 name) with the regressor's other valid options like `absolute_error` and `friedman_mse`. Serving that answer to the classifier question would be confidently wrong — `RandomForestClassifier` accepts `gini` (the default), `entropy`, and `log_loss` as criterion values, and `squared_error` is not even a legal argument.

3. Run the near-miss through `lookup` at three thresholds directly, so you can see the similarity score and the hit-or-miss decision side by side:

   ```
   uv run python -c "
   from src.cache.semantic import lookup, _collection
   from src.embedder import embed_query

   q = 'What is the default criterion for RandomForestClassifier?'
   col = _collection()
   res = col.query(query_embeddings=[embed_query(q)], n_results=1, include=['metadatas', 'distances'])
   distance = res['distances'][0][0]
   similarity = 1.0 - distance
   nearest_q = res['metadatas'][0][0]['question']
   print(f'query: {q}')
   print(f'nearest cached question: {nearest_q}')
   print(f'similarity: {similarity:.4f}')
   for t in [0.70, 0.85, 0.95]:
       hit = lookup(q, threshold=t)
       label = ('HIT  -> answer head: ' + hit.answer[:80].replace(chr(10), ' ')) if hit else 'MISS'
       print(f'  threshold={t}: {label}')
   "
   ```

   Expected output shape (your exact similarity number varies a few hundredths depending on the embedder run, but the boundary structure is reliable):

   ```
   query: What is the default criterion for RandomForestClassifier?
   nearest cached question: What is the default criterion for RandomForestRegressor?
   similarity: 0.8312
   threshold=0.70: HIT  -> answer head: The default criterion for RandomForestRegressor is squared_error ...
   threshold=0.85: MISS
   threshold=0.95: MISS
   ```

   The similarity score in the low 0.83 range against a different scikit-learn estimator is the canonical near-miss pattern: the embedder identifies the shared template (the "what is the default criterion for X" structure) and rates the two queries as semantically close, but the answer that is correct for one is wrong for the other. Module 14 called this out as the failure mode where embedding-similarity-as-cache-key breaks down — Zhu, Zhu, and Jiao's 2024 Berkeley paper makes the formal argument that off-the-shelf semantic embeddings are not optimized for caching prediction, and the same-template-different-estimator case is exactly the family of queries where stock embeddings score similar without sharing a correct answer.

4. Walk the consequences out loud. At the 0.70 threshold the cache served `RandomForestRegressor` content to a question about `RandomForestClassifier`. The learner who reads that answer is told the default criterion is `squared_error` and that the valid alternatives are `absolute_error`, `friedman_mse`, and `poisson` — none of which scikit-learn accepts as a `RandomForestClassifier(criterion=...)` argument. Passing `criterion="squared_error"` to a `RandomForestClassifier` raises `InvalidParameterError` in scikit-learn 1.5 and later. The cache answer is not just imprecise; it is operationally wrong, and a copy-and-paste user would discover the wrongness only when their training script crashed. That answer would not register on a cost dashboard as a failure — the response went out fast, no LLM call was made, no error was logged. The wrong-answer mode is silent. Module 14 named the three-technique tuning loop (offline eval, shadow mode, A/B) that catches this in production; on the development workspace the equivalent is the threshold sweep you just ran, but the framing is the same — the threshold is the operating parameter you tune against your workload, and a careless setting trades savings for silent quality degradation.

5. Optional extension: re-run the near-miss through `cached_route_query` at the default threshold to confirm the production path correctly misses and that the freshly generated answer talks about `gini`, not `squared_error`:

   ```
   uv run python -c "
   from src.cache import cached_route_query
   q = 'What is the default criterion for RandomForestClassifier?'
   r = cached_route_query(q)
   print('cached:', r.cached)
   print('answer head:', r.answer[:160])
   "
   ```

   The response shows `cached: False` and an LLM-produced answer about `gini` (the actual default for `RandomForestClassifier`). The default 0.85 threshold protected the call.

### Acceptance criterion

Three pieces of evidence. First, the captured output of the threshold-sweep on the near-miss query — the similarity score, the nearest cached question, and the hit-or-miss decision at 0.70, 0.85, and 0.95. Second, a one-paragraph interpretation naming what the cache would have served at 0.70, why that answer is wrong for `RandomForestClassifier` (the `squared_error` criterion is not a valid argument for the classifier; the classifier accepts `gini`, `entropy`, `log_loss`), and why the default 0.85 threshold prevented the failure. Third, the through-the-route confirmation that the default threshold correctly missed and returned an LLM-generated answer about the right estimator. The interpretation paragraph is the deliverable that matters — the threshold curve only earns its tuning when you can articulate the wrong-answer mode in your own words.

## Exercise 3 — Cost delta against Module 13's cost log

Exercise 1 showed the cache hits; Exercise 3 puts dollars on them. The cost log from Module 13 records one row per LLM call. Cache hits do not produce rows. The difference between the row count and the request count is the cache's direct dollar value, and the cost log already carries the per-call dollar figure you can use to extrapolate.

### What to do

1. Set up a clean comparison. Move the cost log aside so this exercise's measurement is not contaminated by prior runs, and clear the cache:

   ```
   mv data/cost_log.jsonl data/cost_log.prior.jsonl 2>/dev/null
   uv run python -c "from src.cache import clear; print('removed:', clear())"
   ```

2. Run the fifteen-query paraphrase set from Exercise 1 again. With the cache empty and the cost log fresh, the count of rows added to `data/cost_log.jsonl` will be exactly five — one per cache miss (the first of each group). The other ten queries hit the cache and bypass the LLM, so they do not log. Note that `cached_route_query` calls `run_pipeline`, which records the cost via `compute_cost` from `src/pricing.py:24-30` and (for the live `/query` route the gateway module will wire) `log_request` from `src/cost/tracker.py:38-70`. Module 15's standalone path expects you to call `log_request` yourself on the miss path; the simplest pattern is to use the same fifteen-query loop and tee each miss into the cost log:

   ```
   uv run python -c "
   from src.cache import cached_route_query
   from src.cost.tracker import log_request
   queries = [
       'What is the default criterion for RandomForestRegressor?',
       'Default split criterion in RandomForestRegressor?',
       'RandomForestRegressor default criterion?',
       'How many trees does RandomForestClassifier use by default?',
       'Default n_estimators for RandomForestClassifier?',
       'Number of trees in RandomForestClassifier default?',
       'What kernel does SVC use by default?',
       'Default kernel for SVC?',
       \"SVC's default kernel choice?\",
       'What scoring metric does cross_val_score use by default?',
       'Default scoring for cross_val_score?',
       'cross_val_score default scoring metric?',
       'What is the default test_size for train_test_split?',
       'Default test_size for train_test_split?',
       'train_test_split default test size?',
   ]
   for q in queries:
       r = cached_route_query(q)
       if not r.cached:
           log_request(r.model, r.tokens, r.cost_usd, 'complex')
   print('done')
   "
   wc -l data/cost_log.jsonl
   ```

   Expected: five rows.

3. Sum the cost of those five rows — that is your with-cache spend for the fifteen-query workload:

   ```
   uv run python -c "
   import json
   total = sum(json.loads(line)['cost_usd'] for line in open('data/cost_log.jsonl'))
   print(f'with-cache spend (5 LLM calls + 10 hits): \${total:.6f}')
   "
   ```

   On `gpt-4o` at the ScikitDocs RAG prompt and completion sizes (about 1,900 prompt tokens after the docs context, ~60 completion tokens for a one-paragraph answer), this lands somewhere around $0.025-$0.030 in total — call it two to three cents for the five misses.

4. Now project what those same fifteen queries would have cost without the cache. The simplest defensible estimate is to multiply the five-row average cost by fifteen — the hit queries would each have cost roughly the same as the misses, since the prompt and completion shapes are comparable across paraphrases of the same intent:

   ```
   uv run python -c "
   import json
   rows = [json.loads(l) for l in open('data/cost_log.jsonl')]
   avg = sum(r['cost_usd'] for r in rows) / len(rows)
   without_cache = avg * 15
   with_cache = sum(r['cost_usd'] for r in rows)
   savings = without_cache - with_cache
   print(f'avg cost per LLM call:    \${avg:.6f}')
   print(f'without-cache projection: \${without_cache:.6f} (15 LLM calls)')
   print(f'with-cache actual:        \${with_cache:.6f} (5 LLM calls)')
   print(f'savings:                  \${savings:.6f} ({100*savings/without_cache:.1f}%)')
   "
   ```

   At a 10-of-15 hit rate, the cost reduction lands near 67 percent — exactly the cache-hit ratio. That is the headline number on a paraphrase-heavy workload at the default threshold: hit rate is dollars saved, modulo the embedding-call cost of the lookup (Module 14 named the break-even framing — the embedding call is order-of-magnitude cheaper than the chat completion, so even at 10 percent hit rate the math clears).

5. Extrapolate to a monthly volume. Pick a target — say 10,000 queries per day at the same paraphrase mix — and compute the projected monthly saving:

   ```
   uv run python -c "
   import json
   rows = [json.loads(l) for l in open('data/cost_log.jsonl')]
   avg = sum(r['cost_usd'] for r in rows) / len(rows)
   hit_rate = 10 / 15
   queries_per_day = 10_000
   per_day_no_cache = avg * queries_per_day
   per_day_with_cache = avg * queries_per_day * (1 - hit_rate)
   monthly_savings = (per_day_no_cache - per_day_with_cache) * 30
   print(f'no-cache: \${per_day_no_cache:.2f}/day, \${per_day_no_cache*30:.2f}/month')
   print(f'with-cache: \${per_day_with_cache:.2f}/day, \${per_day_with_cache*30:.2f}/month')
   print(f'monthly savings: \${monthly_savings:.2f}')
   "
   ```

   The dollar amounts are workload-dependent and the hit rate assumption is the load-bearing input — production teams calibrate the projection against measured hit rates from shadow-mode runs, not from a fifteen-query exercise. The point of the projection is the shape of the savings curve: savings scale linearly with volume and with hit rate, sublinearly with the embedding call cost. Module 14 referenced Kostov 2026 and similar practitioner case studies that report layered cost reductions in the 30-to-67 percent range on real workloads when caching is combined with tiered routing and prompt compression; your number here is the caching contribution alone.

### Acceptance criterion

Three numbers and a paragraph. First, the with-cache spend (five LLM calls) for the fifteen-query run. Second, the without-cache projection (fifteen LLM calls) and the absolute and percentage savings. Third, the monthly projection at 10,000 queries per day. The one-paragraph note discusses which inputs to the projection are workload-dependent (the hit rate, the average per-call cost, the volume) and which are stable across deployments (the cosine-distance threshold, the embedder choice). A second paragraph names which knob you would tune in production — the threshold to chase hit rate, the embedder to chase precision, or the TTL to control freshness — and why, based on the exercises you just ran.

## Common pitfalls

A few traps that catch most learners on this module.

**Cache poisoning via prompt injection.** If an attacker successfully prompt-injects the model and you write the injected response to the cache, every paraphrase of that query will be served the bad answer until the entry expires or is evicted. Module 15's `cached_route_query` does not enforce input guards — the gateway module lands the route that wires the input-side prompt-injection check before the cache lookup, and the guardrails module lands the output-side guards that gate the cache *write*. The hardening rule, named forward in those modules, is: never cache a response from an input that tripped a guard, and tag cached entries with their guardrail-pass status so a future policy change can drop them.

**Concurrent write contention.** Chroma's `PersistentClient` serializes writes to its on-disk segment files internally, but the cache's `store` call has a read-then-write pattern (count, embed, upsert) that is not atomic across processes. The starter's single-process Python shell does not need to worry about it; a multi-worker deployment behind gunicorn or uvicorn workers should either route cache writes through one worker or use an external Chroma server (`chromadb.HttpClient`) that owns the write lock. The same architectural exit ramp Module 14 named for scaling — embedded `PersistentClient` for development, hosted Chroma for production — applies here.

**Stale cache after corpus update.** When the underlying corpus changes — a new scikit-learn release, a deprecated API note, a section rewrite — the cache is now serving answers built on outdated retrieval. Module 14 named three invalidation patterns: TTL (the starter's default at 3,600 seconds, suitable for a daily-update corpus), tag-based with `where` filtering, and versioning. The pragmatic first move for any production deployment is to drop the entire cache after a `make load-data` run that mutates the corpus (`uv run python -c "from src.cache import clear; print(clear())"`); the more sophisticated move is to tag each cache entry with the `doc_id`s that contributed to the answer and delete by tag. The RAGOps module builds the blue/green index swap that gives you a cleaner version of the same operation at the retrieval layer; the cache is still your responsibility.

**Mismatched embedders between write and read.** The cache only behaves correctly if the embedder on the write path and the read path are the same model. Swapping `text-embedding-3-small` for `text-embedding-3-large` mid-flight, or A/B-testing two embedders against the same cache collection, silently degrades hit quality because the cosine distances are no longer comparable across the two embedding spaces. The starter enforces single-embedder by routing both paths through `embed_query` at `src/embedder.py:69`; the rule when you change embedders is to clear the cache, retune the threshold from scratch, and treat the dollar projections as new.

**Threshold drift after embedder swap.** Building on the prior point: thresholds are not portable across embedders. The 0.85 baseline that works for `text-embedding-3-small` is not the right number for `text-embedding-3-large` or for a sentence-transformers model; the cosine scale is different, the semantic neighborhood structure is different, and the curve from Exercise 2's threshold sweep will sit at a different inflection point. Re-tune the threshold every time you change the embedder.

**Embedding-call cost as overhead on low-hit-rate workloads.** Every cache lookup, hit or miss, pays for one embedding call. On a paraphrase-heavy docs-FAQ workload the math clears comfortably (the embedding is order-of-magnitude cheaper than the chat completion, so even a 10 percent hit rate pays for itself), but on a workload where queries are diverse and the hit rate sits near zero, the embedding spend is pure overhead. The Module 14 break-even framing applies: measure hit rate before you commit to caching, and treat sub-5 percent hit rates as the diagnostic that the cache is the wrong tool for that workload. The fix is not to lower the threshold and accept more wrong answers; the fix is to instrument the hit rate and turn the cache off if it does not earn its keep on your traffic mix.

## What you have now

A fifteen-query hit-rate report at the default threshold (rubric §10 meets-spec evidence, with the paraphrase set sized larger than the rubric's six-query floor so the signal is statistically meaningful). A near-miss case demonstrating threshold sensitivity with a concrete wrong-answer example (rubric §10 stretch — the sweep at 0.70 / 0.85 / 0.95 with a false-positive call-out at the loose threshold). A cost-delta artifact pairing the cache hits with Module 13's cost log, plus a monthly-volume projection for production planning. Together those three artifacts cover the cache's full operational surface: hit rate as the savings driver, threshold as the quality knob, and dollar volume as the planning unit.

Two forward references. The gateway module builds the HTTP route and moves `cached_route_query`'s composition inline at the route boundary — caching and tiered routing compose multiplicatively, and the cost stack you now have is the substrate both layers run on. The guardrails module picks up the safety side and the cache-poisoning concern Module 14 named — the no-cache-on-failed-guard rule is what that module wires into the gateway, and the exercises that explore output guardrails will revisit the cache write seam to confirm the safe-default holds across guard policies. The cache you built today is upstream of every one of those decisions.
