---
module_number: 15
module_title: "Implement Semantic Caching with Chroma"
slug: semantic-caching
unit_type: overview-video-script
target_minutes: "5-7"
word_count: 642
---

# Module 15 Overview Video: Semantic Caching with Chroma

> Recording aid for a 5-7 min overview (read pace ~100 wpm). Read all prose aloud, top to bottom. `*[Stage: ...]*` lines are screen directions, not spoken. Not learner-facing.

## 0. VS Code setup (before you hit record)

Open the files you will show on camera (run from the repo root):

```bash
code exercises/15-semantic-caching/semantic-caching-starter/DEMO.md exercises/15-semantic-caching/semantic-caching-starter/INSTRUCTIONS.md exercises/15-semantic-caching/semantic-caching-starter/INTERFACES.md exercises/15-semantic-caching/semantic-caching-starter/src/cache/semantic.py exercises/15-semantic-caching/semantic-caching-starter/src/cache/wrapper.py exercises/15-semantic-caching/semantic-caching-starter/src/embedder.py exercises/15-semantic-caching/semantic-caching-starter/src/cost/tracker.py exercises/15-semantic-caching/semantic-caching-starter/src/pricing.py
```

Files and why each is on screen:
- `DEMO.md`: the walkthrough that warms the cache, fires a paraphrase, and sweeps the threshold; reference it as you talk.
- `INSTRUCTIONS.md`: the three write-up exercises; keep it open to point at each acceptance criterion.
- `INTERFACES.md`: the frozen signatures the cache reads through, including `run_pipeline` and `embed_query`.
- `src/cache/semantic.py`: the whole cache in about 140 lines; the threshold gate and the distance-to-similarity line live here.
- `src/cache/wrapper.py`: `cached_route_query`, the entry point all three exercises drive.
- `src/embedder.py`: the shared `embed_query` function; same model on write and read is what makes the scores comparable.
- `src/cost/tracker.py`: `log_request`, which Exercise 3 calls on the miss path to record a cost row.
- `src/pricing.py`: `compute_cost`, the per-call dollar figure the cost delta extrapolates from.

## 1. Intro  (~45-60s)

*[Stage: DEMO.md on screen.]*

In this module you study a semantic cache: a layer that answers a repeat question without paying for the model again. It embeds each query, searches for the nearest cached question, and returns that answer when they're close enough.

There's no new code to write here. The cache ships complete. Your job is to run real queries through it and judge its behavior. You'll measure a hit rate, build a query that fools it, and turn the hits into dollars saved.

## 2. Topic overview  (~60-90s)

*[Stage: open src/cache/semantic.py, point to the threshold gate.]*

A semantic cache has four moves: embed the query, search the cache, return on a hit, or call the model and write back on a miss. One number decides everything: the similarity threshold. The default here is zero point eight five.

Here's the mental model. Similarity is the cosine between two question vectors. One means identical wording, zero means unrelated. A hit means the new question reads close to a stored one.

And here's the misconception this whole module attacks. Reading alike is not the same as having the same answer. A one-word swap can read almost identical yet need a completely different answer. So a raw cosine threshold alone can't safely gate a cache.

One setup note that's easy to miss. After `make setup` and `make load-data`, run `make seed-difficulty`. That step plants the facts the demo's near-miss leans on. Skip it and the cache has nothing real to serve.

## 3. Exercise call-outs

### Exercise 1: Fifteen paraphrases, hit-rate report

*[Stage: switch to INSTRUCTIONS.md, Exercise 1.]*

The first exercise fires fifteen queries through `cached_route_query`: five base questions, each with two paraphrases. The first time each base question arrives, the cache is empty for it, so it misses and calls the model. The two paraphrases that follow should hit that write.

Here's what to watch for. The expected pattern is five misses and ten hits, about sixty-seven percent. If a paraphrase is reworded too hard, its score can slip under zero point eight five and miss. That's normal. You're done with a fifteen-row hit-or-miss table and a cache showing exactly five stored originals.

### Exercise 2: The near-miss case

*[Stage: scroll to Exercise 2; keep the threshold-sweep output in view.]*

This is the heart of the module. You ask about a different estimator, the classifier instead of the regressor, one word away from a cached question. You'd expect the threshold to reject it. It doesn't.

Here's the watch-out. That near-miss scores about zero point nine six, higher than several of your genuine paraphrases. So it hits at every threshold you try, and serves the regressor's answer to a classifier question. No threshold both keeps your paraphrases and blocks it. The fix isn't tuning the number. It's a layer up: a re-ranker, a metadata filter on the estimator, or a meaning-aware cache key. You're done when you've named that fix in your own words.

### Exercise 3: Cost delta against the cost log

*[Stage: point to log_request in src/cost/tracker.py, then compute_cost in src/pricing.py.]*

The third exercise puts dollars on the hits. Hits skip the model, so they never write a cost row. You rerun the fifteen queries, tee each miss into the cost log, and you'll get exactly five rows.

Here's the framing to hold. Sum those five, then project what fifteen calls would have cost. At a ten-of-fifteen hit rate, savings land near sixty-seven percent: the hit rate is the dollars saved. Then scale it to ten thousand queries a day. You're done with three numbers and a note on which inputs are workload-dependent, like hit rate, and which are stable.

## 4. Key insights  (~30-45s)

*[Stage: return to the threshold gate in src/cache/semantic.py.]*

Three takeaways. First, similarity is not task relevance: a near-miss can outscore a true paraphrase, so cosine alone can't gate a cache safely. Second, the fix for that is architectural, a re-ranker or a meaning-aware key, never just nudging the threshold. Third, hit rate is the savings driver, so measure it before you trust the cache to earn its keep.

## 5. Outro  (~20-30s)

*[Stage: return to DEMO.md.]*

That's the semantic cache: four moves, one threshold, and one sharp limit. The next module wires this composition into the HTTP gateway, where caching and routing stack together. The guardrails work then closes the cache-poisoning gap. See you in the next one.
