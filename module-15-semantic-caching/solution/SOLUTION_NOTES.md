# Solution notes — Module 15 (Semantic Caching)

The starter already ships the complete cache stack — `src/cache/semantic.py` (`lookup`, `store`, `clear`), `src/cache/wrapper.py` (`cached_route_query`), and `src/cache/__init__.py` (public API). **There are no new files to author for this module.** All three exercises are analysis runs whose deliverables are writeup artifacts (output captures, interpretation paragraphs, and a monthly cost projection). These notes document the expected shape of each artifact.

## Files added

None. The cache primitives are pre-wired in the starter; the exercises measure their behavior.

## Writeup-only deliverables (no code to commit)

### Exercise 1 — Fifteen paraphrases, hit-rate report

Expected paste-into-writeup shape after running the fifteen-query loop through `cached_route_query`:

```
False | What is the default criterion for RandomForestRegressor?
True  | What's the default criterion used by RandomForestRegressor?
True  | What is the default value of the criterion parameter in RandomForestRegressor?
False | What is the default kernel for SVC?
True  | What is the default kernel used by SVC?
True  | What's the default kernel for the SVC classifier?
False | What is the default test_size for train_test_split?
True  | What's the default test_size used by train_test_split?
True  | What is the default value of test_size in train_test_split?
False | What is the default scoring metric for cross_val_score?
True  | What's the default scoring metric in cross_val_score?
True  | What is the default scoring used by cross_val_score?
False | What is the default init method for KMeans?
True  | What is the default initialization method used by KMeans?
True  | What's the default init for KMeans?
hits=10/15 (66.7%)
```

Expected cache-contents dump (five entries — one original per base question):

```
cache entries: 5
  - What is the default criterion for RandomForestRegressor?
  - What is the default kernel for SVC?
  - What is the default test_size for train_test_split?
  - What is the default scoring metric for cross_val_score?
  - What is the default init method for KMeans?
```

The one-paragraph interpretation should name the 10/15 result as the expected pattern (each first-arrival misses and writes; the two following paraphrases per group hit the original write), flag any paraphrase that missed unexpectedly (the most common case is an aggressively reworded paraphrase whose cosine slipped below 0.85), and note that hit rates in the 60-75% range on a paraphrase-heavy set are within the workload envelope Regmi and Pun 2024 reported on customer-service traffic. On `text-embedding-3-small` the chosen paraphrases land between roughly 0.86 and 0.99 against their base — comfortably above the 0.85 threshold — which is why this set reproduces 10/15 cleanly. Exercise 2 then shows the unsettling flip side: a non-paraphrase that scores higher than several of these.

### Exercise 2 — Near-miss threshold sweep

Expected output from the `lookup` sweep at 0.70 / 0.85 / 0.95 against the cached "What is the default criterion for RandomForestRegressor?" entry:

```
query: What is the default criterion for RandomForestClassifier?
nearest cached question: What is the default criterion for RandomForestRegressor?
similarity: 0.9563
threshold=0.70: HIT  -> answer head: The default criterion for `sklearn.ensemble.RandomForestRegressor` is `'squared_
threshold=0.85: HIT  -> answer head: The default criterion for `sklearn.ensemble.RandomForestRegressor` is `'squared_
threshold=0.95: HIT  -> answer head: The default criterion for `sklearn.ensemble.RandomForestRegressor` is `'squared_
```

`text-embedding-3-small` is deterministic, so the similarity reproduces to four places at 0.9563 — a hit at all three thresholds, including the strict 0.95. This is the central result: the one-word `Regressor`→`Classifier` swap is *more* similar (0.9563) than three of the ten genuine Exercise 1 paraphrases (which land at 0.8622, 0.9003, and 0.9448). No threshold both admits the real paraphrases and rejects this near-miss.

The interpretation paragraph should make these points: (1) the near-miss similarity (0.9563) sits inside — and near the top of — the Exercise 1 paraphrase band (0.86-0.99), so it outscores several correct paraphrases; (2) the cache served regressor content (`squared_error`, with the regressor's valid alternatives `absolute_error`, `friedman_mse`, `poisson`) to a classifier question, at the default 0.85 and even at 0.95; (3) `RandomForestClassifier` accepts `gini` (default), `entropy`, and `log_loss` — `squared_error` is not a valid argument and `criterion="squared_error"` raises `InvalidParameterError` in scikit-learn 1.5+; (4) the wrong-answer mode is silent (no LLM call, no error logged, no cost-dashboard signal); (5) the fix is architectural — a re-ranker / cross-encoder that scores question-to-question relevance, a metadata filter on the estimator class, or a criterion-aware cache key — not a threshold tweak, because surface similarity and task relevance are different axes and raw cosine on this embedder cannot separate them.

Optional extension — `cached_route_query` at the default threshold prints `cached: True` and serves the cached `squared_error` (regressor) answer to the classifier question. The default 0.85 did *not* protect the call. To see the correct `gini` answer, clear the cache and re-run: with an empty cache the query misses, the LLM is called, and the fresh answer names `gini`. The only thing that produced a right answer was an empty cache, not the threshold.

### Exercise 3 — Cost delta and monthly projection

Expected with-cache spend output (your exact numbers depend on `gpt-4o` token sizes for the day):

```
with-cache spend (5 LLM calls + 10 hits): $0.029220
```

Lands in the $0.025-$0.030 band on `gpt-4o` at ~1,900 prompt tokens + ~60 completion tokens per call. Exact cents move with the day's token sizes; the band and the structure are what matter.

Expected savings comparison (numbers from a representative live run; yours will be within a few percent):

```
avg cost per LLM call:    $0.005844
without-cache projection: $0.087660 (15 LLM calls)
with-cache actual:        $0.029220 (5 LLM calls)
savings:                  $0.058440 (66.7%)
```

The percentage savings tracks the cache-hit ratio exactly (10/15 = 66.7%) regardless of the absolute dollar figures, modulo the embedding-call overhead, which is order-of-magnitude smaller than the chat-completion cost — the cache pays for itself at any non-trivial hit rate.

Expected monthly projection at 10,000 queries/day:

```
no-cache:   $58.44/day, $1753.20/month
with-cache: $19.48/day, $584.40/month
monthly savings: $1168.80
```

The interpretation paragraph should split the projection inputs into workload-dependent (hit rate, per-call cost, volume) and stable-across-deployments (cosine threshold, embedder choice). The second paragraph should name which knob the learner would tune in production. The honest answer after Exercise 2 is layered: the threshold is the knob for *paraphrase recall* (raise it for precision, lower it for hit rate), but it is the wrong tool for the near-miss wrong-answer mode — that needs a re-ranker, a metadata filter, or a criterion-aware key. The TTL is the freshness knob (tune per corpus update cadence); the embedder is a higher-cost change (forces a threshold retune and a cache clear). A strong answer names the threshold for the recall/precision trade *and* concedes that chasing the near-miss with the threshold makes things worse, pointing at the architectural fix instead.

## Verification

No script to run. The deliverables are output captures and writeup paragraphs. Run the three exercise loops from `INSTRUCTIONS.md` against a fresh cache and a fresh `data/cost_log.jsonl` to reproduce the expected outputs.

If you want a sanity check that the cache module imports cleanly without `make load-data`:

```bash
uv run python -c "from src.cache import lookup, store, clear, cached_route_query, COLLECTION_NAME; print('OK', COLLECTION_NAME)"
```

Should print `OK cache`.

## KNOWN-LIMITATIONs

None. This module's exercises are pure measurement against a pre-wired cache stack; no code in the starter needs to be modified to complete them. The single `src/cache/` module ships unchanged.
