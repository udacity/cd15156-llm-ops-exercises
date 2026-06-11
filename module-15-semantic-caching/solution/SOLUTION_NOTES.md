# Solution notes — Module 15 (Semantic Caching)

The starter already ships the complete cache stack — `src/cache/semantic.py` (`lookup`, `store`, `clear`), `src/cache/wrapper.py` (`cached_route_query`), and `src/cache/__init__.py` (public API). **There are no new files to author for this module.** All three exercises are analysis runs whose deliverables are writeup artifacts (output captures, interpretation paragraphs, and a monthly cost projection). These notes document the expected shape of each artifact.

## Files added

None. The cache primitives are pre-wired in the starter; the exercises measure their behavior.

## Writeup-only deliverables (no code to commit)

### Exercise 1 — Fifteen paraphrases, hit-rate report

Expected paste-into-writeup shape after running the fifteen-query loop through `cached_route_query`:

```
False | What is the default criterion for RandomForestRegressor?
True  | Default split criterion in RandomForestRegressor?
True  | RandomForestRegressor default criterion?
False | How many trees does RandomForestClassifier use by default?
True  | Default n_estimators for RandomForestClassifier?
True  | Number of trees in RandomForestClassifier default?
False | What kernel does SVC use by default?
True  | Default kernel for SVC?
True  | SVC's default kernel choice?
False | What scoring metric does cross_val_score use by default?
True  | Default scoring for cross_val_score?
True  | cross_val_score default scoring metric?
False | What is the default test_size for train_test_split?
True  | Default test_size for train_test_split?
True  | train_test_split default test size?
hits=10/15 (66.7%)
```

Expected cache-contents dump (five entries — one original per base question):

```
cache entries: 5
  - What is the default criterion for RandomForestRegressor?
  - How many trees does RandomForestClassifier use by default?
  - What kernel does SVC use by default?
  - What scoring metric does cross_val_score use by default?
  - What is the default test_size for train_test_split?
```

The one-paragraph interpretation should name the 10/15 result as the expected pattern (each first-arrival misses and writes; the two following paraphrases per group hit the original write), flag any paraphrase that missed unexpectedly (the most common case is a 0.83-cosine slip below the 0.85 threshold — Exercise 2 is the deliberate construction of that case), and note that hit rates in the 60-75% range on a paraphrase-heavy set are within the workload envelope Regmi and Pun 2024 reported on customer-service traffic.

### Exercise 2 — Near-miss threshold sweep

Expected output from the `lookup` sweep at 0.70 / 0.85 / 0.95 against the cached "What is the default criterion for RandomForestRegressor?" entry:

```
query: What is the default criterion for RandomForestClassifier?
nearest cached question: What is the default criterion for RandomForestRegressor?
similarity: 0.8312
threshold=0.70: HIT  -> answer head: The default criterion for RandomForestRegressor is squared_error ...
threshold=0.85: MISS
threshold=0.95: MISS
```

The exact similarity number varies a few hundredths between embedder runs (typical band: 0.81-0.84); the boundary structure is reliable — hit at 0.70, miss at 0.85 and 0.95.

The interpretation paragraph should make three points: (1) at threshold 0.70 the cache served regressor content (`squared_error`, `absolute_error`, `friedman_mse`, `poisson` — the regressor's valid criterion arguments) to a classifier question; (2) `RandomForestClassifier` accepts `gini` (default), `entropy`, and `log_loss` — `squared_error` is not a valid argument and `criterion="squared_error"` raises `InvalidParameterError` in scikit-learn 1.5+; (3) the wrong-answer mode is silent (no LLM call, no error logged, no cost-dashboard signal), which is exactly why the threshold is a quality knob first and a cost knob second.

Optional extension — `cached_route_query` at the default threshold should print `cached: False` and an LLM-produced answer mentioning `gini` (the actual classifier default). The default 0.85 protected the call.

### Exercise 3 — Cost delta and monthly projection

Expected with-cache spend output (your exact numbers depend on `gpt-4o` token sizes for the day):

```
with-cache spend (5 LLM calls + 10 hits): $0.027500
```

Lands in the $0.025-$0.030 band on `gpt-4o` at ~1,900 prompt tokens + ~60 completion tokens per call.

Expected savings comparison:

```
avg cost per LLM call:    $0.005500
without-cache projection: $0.082500 (15 LLM calls)
with-cache actual:        $0.027500 (5 LLM calls)
savings:                  $0.055000 (66.7%)
```

The percentage savings tracks the cache-hit ratio exactly (10/15 = 66.7%), modulo the embedding-call overhead, which is order-of-magnitude smaller than the chat-completion cost — the cache pays for itself at any non-trivial hit rate.

Expected monthly projection at 10,000 queries/day:

```
no-cache:   $55.00/day, $1650.00/month
with-cache: $18.33/day, $549.99/month
monthly savings: $1100.01
```

The interpretation paragraph should split the projection inputs into workload-dependent (hit rate, per-call cost, volume) and stable-across-deployments (cosine threshold, embedder choice). The second paragraph should name which knob the learner would tune in production — the most defensible answer is the threshold, because it directly trades hit rate against quality and the wrong-answer mode demonstrated in Exercise 2 is the operational risk that bounds how aggressively it can be lowered. The TTL is the freshness knob (tune per corpus update cadence); the embedder is a higher-cost change (forces a threshold retune and a cache clear).

## Verification

No script to run. The deliverables are output captures and writeup paragraphs. Run the three exercise loops from `INSTRUCTIONS.md` against a fresh cache and a fresh `data/cost_log.jsonl` to reproduce the expected outputs.

If you want a sanity check that the cache module imports cleanly without `make load-data`:

```bash
uv run python -c "from src.cache import lookup, store, clear, cached_route_query, COLLECTION_NAME; print('OK', COLLECTION_NAME)"
```

Should print `OK cache`.

## KNOWN-LIMITATIONs

None. This module's exercises are pure measurement against a pre-wired cache stack; no code in the starter needs to be modified to complete them. The single `src/cache/` module ships unchanged.
