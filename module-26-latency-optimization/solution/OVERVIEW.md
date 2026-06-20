---
module_number: 26
module_title: "Optimize End-to-End RAG Latency with Streaming and Profiling"
slug: latency-optimization
unit_type: overview-video-script
target_minutes: "5-7"
word_count: 685
---

# Module 26 Overview Video: Latency Optimization

> Recording aid for a 5-7 min overview (read pace ~100 wpm). Read all prose aloud, top to bottom. `*[Stage: ...]*` lines are screen directions, not spoken. Not learner-facing.

## 0. VS Code setup (before you hit record)

Open the files you will show on camera:

- `../latency-optimization-starter/DEMO.md`
- `../latency-optimization-starter/INSTRUCTIONS.md`
- `../latency-optimization-starter/INTERFACES.md`
- `../latency-optimization-starter/src/tracing.py`
- `../latency-optimization-starter/src/streaming.py`
- `../latency-optimization-starter/src/store.py`

Files and why each is on screen:
- `DEMO.md`: the walkthrough you reference; it traces the pipeline, the streaming route, and the HNSW knob location end to end.
- `INSTRUCTIONS.md`: the three exercises; keep it open to point at each success criterion.
- `INTERFACES.md`: the frozen signatures, including the streaming endpoint contract. This module profiles existing code, so nothing here changes.
- `src/tracing.py`: where `init_tracing` boots Phoenix and `traced_pipeline` emits the retrieval and generator spans Exercise 1 reads.
- `src/streaming.py`: the Server-Sent Events route Exercise 2 measures; the `stream_options` flag and the guards seam live here.
- `src/store.py`: the `get_collection` function with the one HNSW line that is set and the knobs that are left at defaults.

## 1. Intro  (~45-60s)

*[Stage: DEMO.md on screen.]*

This is the last implementation module, and you won't add a single new component. Everything is already wired: the pipeline, the cache, the gateway, the streaming route, and Phoenix tracing. Your job is to profile what's already running and decide what's worth optimizing.

By the end you'll have three artifacts: a cold-versus-cached latency table, a time-to-first-token comparison, and a vector-search sweep. The real skill here isn't tuning. It's measuring first, so you tune the thing that actually matters.

## 2. Topic overview  (~60-90s)

*[Stage: bring src/tracing.py forward.]*

Latency optimization starts with one habit: profile before you tune. You can't fix what you haven't measured.

Phoenix is the tool you'll use. It's an in-process tracing UI that runs inside the same worker as the app, on port 6006. Every step of a query emits a span, and you read those spans to see where the wall-clock time actually went. If port 6006 isn't reachable in your workspace, `make show-traces` prints the same data as a table.

Here's the one idea to hold onto, and it's the misconception that trips up everyone. Time-to-first-token, or TTFT, is not the same metric as total time. They move independently. Streaming can make the first token appear almost instantly while the total time barely changes. So you always measure both, because they answer two different questions: how fast does it feel, and how long does it really take.

## 3. Exercise call-outs

### Exercise 1: Profile one query, build the cold-versus-cached table

*[Stage: switch to INSTRUCTIONS.md, Exercise 1.]*

The first exercise fires the same question five times. The first call misses the cache and walks the full pipeline. The next four hit. You read the Phoenix traces and record four span durations: the cache lookup, the classifier, retrieval, and the generator.

Here's what to watch out for. A cache hit feels nearly free, but it isn't. The lookup still makes one embedding round-trip to the hosted endpoint before it can match anything. So the speedup is roughly an order of magnitude on a fast endpoint, not a hundred times, and not down to a few milliseconds. On the Vocareum proxy, where that round-trip dominates, the gap is only a few-fold. To complete this exercise, produce a two-column table, cold beside cached, and the speedup ratio.

### Exercise 2: Measure time-to-first-token on the streaming endpoint

*[Stage: open src/streaming.py; point to the stream_options line and the guards docstrings.]*

The second exercise times both routes on the same fresh question. You'll measure TTFT and total for the blocking route and for the streaming route, then build a two-by-two table.

Here's the heart of it, and the watch-out. Streaming improves time-to-first-token, not total time. The model still has to finish generating, so the totals come out comparable. Do not write "streaming is several times faster," because that mixes two metrics. The honest framing is that streaming kills the spinner: the user sees words within a few hundred milliseconds even when the full answer takes a few seconds. To complete this exercise, fill in the two-by-two table so that streaming TTFT lands well below the blocking total — and both totals sit close to each other, confirming that streaming buys perceived speed, not throughput.

### Exercise 3: Sweep ef_search against the scikit_docs collection

*[Stage: open src/store.py and point to get_collection, the hnsw:space line.]*

The third exercise builds three sandbox collections with different `ef_search` values and times the vector search alone. This is the query-time knob that trades latency for recall.

Here's what to watch out for, and it's the whole lesson. At the starter's roughly 750 chunks, the differences are tiny, down near the noise floor. The ordering reproduces, so higher `ef_search` is slower, but the magnitude is a fraction of a millisecond and runs can even reorder. That flatness isn't a defect. It's the signal that tuning this knob here wouldn't pay off. At a few million rows the curve sharpens and the trade becomes real. To complete this exercise, build a table of `ef_search` against latency and recall, plus a note on when this knob earns its keep.

## 4. Key insights  (~30-45s)

*[Stage: return to DEMO.md.]*

Three takeaways. First, profile before you tune, because the span that dominates the budget is usually the generator, not the knob you can reach. Second, TTFT and total time are separate metrics, so measure both and frame streaming as a perceived-latency win. Third, effort follows the numbers: cache hit rate and model choice first, vector search later.

## 5. Outro  (~20-30s)

That's the profile-streaming-tune triangle, measured on the exact code the starter ships. You've now closed out the implementation track. The capstone pulls every layer together: retrieval, generation, caching, guards, cost, and the latency discipline you just practiced. See you there.
