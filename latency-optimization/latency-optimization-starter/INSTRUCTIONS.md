# Module 26 — Optimize End-to-End RAG Latency with Streaming and Profiling

## Setup

This starter is the ScikitDocs RAG app — a Q&A assistant for the scikit-learn library — with the full instrumentation stack already wired: prompt loader and Jinja templates, Chroma vector store, RAG pipeline (`run_pipeline`), Phoenix tracing (`init_tracing` + `traced_pipeline`), RAGAS evaluation harness, cost monitoring (`src/pricing.py` + `src/cost/`), semantic answer cache (`src/cache/semantic.py`), FastAPI gateway with tier classifier (`src/gateway/`), guardrails seam (`_pre_stream_guards` shim + blocking-path hooks), A/B testing (`scripts/ab_analyze.py` + `prompts/docbot_system_A.j2`/`_B.j2`), RAGOps watcher (`scripts/start_watcher.py`), and a streaming SSE endpoint (`src/streaming.py`). This is the last implementation module — every prior module's wiring is in place. In this module you will not add new components; you will profile what is already running. The three exercises produce a cold-versus-cached latency table from Phoenix spans, a time-to-first-token comparison between the blocking and streaming endpoints, and an `ef_search` sweep that confirms in your own numbers how the vector-search recall-versus-latency curve behaves at the starter's three-to-four-thousand-chunk scale.

Bring up the corpus before you start:

```bash
uv sync
cp .env.example .env          # add your OPENAI_API_KEY (or Vocareum voc- key)
make load-data                # ~45–60s cold, ~5s warm; ~$0.10 in embeddings
```

Smoke-check the pipeline before any latency work:

```bash
uv run python -c "from src.pipeline import run_pipeline; print(run_pipeline('What kernel does SVC use by default?').answer[:80])"
```

If that returns a grounded answer about `rbf`, you are ready. Follow the demo walkthrough first, then work through the three exercises in order.

---

# Demo — Profile a Query, Stream the Answer, See Where the HNSW Knobs Live

The previous concept module named the principle — profile before you optimize, time-to-first-token and total latency are different metrics, vector-search tuning is small at small scale. This demo brings all three down to the ScikitDocs starter. You will fire one query through the live gateway, open the Phoenix trace it produced, walk the streaming endpoint that ships with the starter, and look at the place in `src/store.py` where the HNSW knobs would go if the corpus grew to ten million rows. Nothing in this demo modifies the capstone. The point is to read what is already there with the right vocabulary.

## Part 1 — Profile with Phoenix: where the time actually goes

Start the starter in one terminal:

```
make load-data    # one-time, if you have not run it yet
make serve
```

The lifespan handler at `src/gateway/app.py:30-38` does two things in the same process — it boots FastAPI on port 8080 and it calls `init_tracing()` from the tracing module (`src/tracing.py`), which launches the embedded Phoenix UI on port 6006 and registers the OpenTelemetry tracer provider. From the moment the `Uvicorn running on http://0.0.0.0:8080` line appears, every span the pipeline emits lands in Phoenix's in-process store. No external service, no API key.

In a second terminal, fire one query against the blocking route:

```
curl -s -X POST http://localhost:8080/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the default value of n_estimators in RandomForestClassifier?"}' | jq
```

The response includes a `trace_id` field. Open `http://localhost:6006` in your browser, navigate to the `scikitdocs` project, find the newest trace, and click in. If port 6006 is not reachable from your browser — common on locked-down workspaces — run `make show-traces` instead. That target shells out to `scripts/show_traces.py` and prints the same data as a markdown table. The rubric §7 evidence path expects either form.

Read the trace top to bottom. The root span is `rag_query`. Its children, roughly in time order, are the layers `route_query` at `src/gateway/router.py:38-76` composes. Treat the magnitudes below as workload- and region-dependent — the shape is the point, not the numbers.

The tier classifier runs first. The two-label JSON-mode prompt at `prompts/classifier.j2` goes through `gpt-4o-mini` and returns either `"simple"` or `"complex"`. On hosted endpoints this is typically in the few-hundred-millisecond range. The gateway module covered the classifier internals; the trace surfaces it as one OpenInference child span emitted by the `OpenAIInstrumentor` that the tracing module wired into `init_tracing`.

The cache lookup is next. On a miss, that is one embedding call to `text-embedding-3-small` plus a Chroma nearest-neighbor query against the `cache` collection — single-digit to low-tens of milliseconds combined. On a hit, the entire downstream pipeline is skipped and the response returns in a few tens of milliseconds total. The semantic-caching modules anchored the cache; the threshold gate is `src.constants.CACHE_SIMILARITY_THRESHOLD = 0.85`.

The retrieval span comes from `traced_pipeline` at `src/tracing.py`. One embedding call for the question, one Chroma similarity search against the `scikit_docs` collection. At the starter's roughly three-to-four-thousand chunks, that lands in the single-digit-millisecond range. The vector-databases module covered the HNSW algorithm; here it is a small constant.

The generator span is where the wall-clock budget actually sits. `OpenAIInstrumentor` wraps the chat completions call and emits an OpenInference span carrying input messages, output messages, token counts, and the model name. Total LLM time on a hosted endpoint typically runs on the order of one to a few seconds — TTFT in the few-hundred-millisecond range, generation at roughly thirty to eighty tokens per second per Anyscale's 2023 benchmark, total length dominated by completion size.

Output guards do not run on the blocking path the starter ships today. The capstone wires a hallucination judge and an off-topic check that fire after generation; the guardrails module walks those scanners against the same gateway and explains the latency they add. For this demo, the blocking `/query` route runs classifier → cache → retrieve → generate and returns. That is the surface Exercise 1 profiles.

Stack the numbers. On a cache miss, the classifier and the generator dominate the wall-clock latency, with the rest distributed across cache lookup, retrieval, and overhead. That distribution is what should drive optimization effort. The cache is the biggest single win on paraphrase-heavy traffic because a hit eliminates the classifier, the retrieval, and the generator spans in one move. The previous concept module named this; Phoenix shows it.

## Part 2 — Streaming the answer with `/query/stream`

The starter ships two query endpoints. The blocking `/query` runs the full pipeline and returns a single JSON body — that is what Part 1 traced. The streaming `/query/stream` is mounted from a separate file at `src/streaming.py` and emits Server-Sent Events token by token. Both routes are wired in by `create_app` at `src/gateway/app.py:42-51`, which calls `application.include_router(streaming_router)` alongside the blocking router and the cost-monitoring module's dashboard.

Read `src/streaming.py` first. The function that does the work is `stream_completion` at the middle of the file. The signature returns `Iterator[str | tuple[str, TokenUsage, float]]` — each iteration yields a token as a string, and the final iteration yields a three-tuple of the assembled answer, the token usage, and the dollar cost. The key API choice is the `stream_options` argument:

```python
stream_options={"include_usage": True},
```

Without that flag, the OpenAI streaming response does not include the `usage` field on its final chunk. The cost computation that follows — `compute_cost(model, usage)` — has nothing to work from, and the cost dashboard reports zero. This is the gotcha the OpenAI cookbook calls out and the starter codes around explicitly. If you copy this pattern to another project, the include-usage option is the line you must not drop.

Now walk the SSE route. `query_stream` at the bottom of `src/streaming.py` wires `_stream` into FastAPI's `StreamingResponse` with `media_type="text/event-stream"`. The input-guards seam is `_pre_stream_guards(question)` — a thin function that today returns `(question, None)` and forward-references the guardrails module as the consumer. When the guardrails module lands, the shim will run the LLM Guard prompt-injection scanner and the Presidio PII redactor before any token is emitted; a prompt-injection match short-circuits to `_blocked_stream` and the response carries exactly one `done` SSE event with `blocked_by` set, and PII in the question is rewritten in place before retrieval and generation. Output guards over a streamed response are intentionally deferred. The hallucination judge and off-topic check both need the whole answer to make a decision, and applying them on a half-finished stream is the streaming-guards composition problem the concept module described. The capstone takes the position that on this workload, the cleaner answer is to either run the full guards on the blocking route or defer them as a follow-up on the streaming route. Both choices live in the codebase so the trade-off is visible.

Test it from terminal two. `-N` disables curl's output buffering so tokens print as they arrive:

```
curl -N -X POST http://localhost:8080/query/stream \
  -H 'Content-Type: application/json' \
  -d '{"question": "How do I tune n_estimators in scikit-learn?"}'
```

You should see `data: {"type": "token", ...}` frames stream in over a few seconds, followed by a final `data: {"type": "done", "response": {...}}` frame carrying the assembled answer, the sources, the model, and the cost. Time-to-first-byte sits in the few-hundred-millisecond range on a cache miss against a hosted model. Total time is similar to the blocking route. The user-perceived difference is the spinner that did not appear, not the total clock. One implementation note: the streaming route deliberately bypasses the semantic answer cache because the cache stores a fully realized `QueryResponse`, not a token sequence. Cache wins are measured on the blocking route in Exercise 1.

## Part 3 — Where the HNSW knobs would live

Open `src/store.py`. The whole file is about one hundred and fifty lines. The relevant block is `get_collection` near the middle:

```python
def get_collection(name: str = "scikit_docs") -> Any:
    return _client().get_or_create_collection(
        name=_resolve_alias(name),
        metadata={"hnsw:space": "cosine"},
    )
```

The teachable surface is the metadata dict. The `scikit_docs` collection pins `hnsw:space=cosine` because OpenAI embeddings are normalized and Chroma's L2 default silently corrupts ranking against them — that override is load-bearing. Beyond `hnsw:space`, none of the other HNSW parameters are set. Chroma's defaults take over: `hnsw:M = 16`, `hnsw:construction_ef = 100`, `hnsw:search_ef = 100`. Compare against the cache collection at `src/cache/semantic.py`, which also pins `hnsw:space=cosine` for the same normalisation reason. Both collections do the one thing that has to be set; both leave the recall-versus-latency knobs at defaults.

If you wanted to tune the docs collection — at scale where it would help — the Chroma syntax is what the docs show:

```python
return _client().get_or_create_collection(
    name=_resolve_alias(name),
    metadata={
        "hnsw:space": "cosine",
        "hnsw:M": 16,
        "hnsw:construction_ef": 200,
        "hnsw:search_ef": 100,
    },
)
```

`hnsw:M` and `hnsw:construction_ef` are build-time parameters. Changing them on an existing collection has no effect — Chroma builds the graph once at insert time. The query-time knob that lets you trade latency for recall after the fact is `hnsw:search_ef`. Higher values explore more candidate neighbors before returning the top-k, which raises recall and raises per-query latency.

The honest framing. At the starter's three-to-four-thousand chunks the curve is measurable but not dramatic — single-digit to low-tens of milliseconds across the typical sweep range. The variance is more visible here than at the capstone's thirty-product scale because more chunks mean more navigation work for the algorithm. At a few million the tuning becomes operationally meaningful and the recall-versus-latency trade becomes a real engineering decision. The discipline transfers regardless of scale: profile before you tune, and tune the knob whose curve sits above the noise floor on your specific workload. The exercise that follows lets you run a sweep on the starter's corpus and see for yourself.

One forward reference. The other latency lever at the model layer — running inference yourself on a GPU — is the vLLM concept module's territory. When the API-hosted model is the bottleneck and you have sustained volume to justify the operational load, self-hosting opens up batching and KV-cache tuning that hosted endpoints do not expose. Most teams do not reach that point. Knowing it exists is the takeaway.

---

# Exercise — Profile a Trace, Stream a Token, Sweep `ef_search`

Three exercises. The first reads one Phoenix trace and turns the spans into a latency budget — cold path versus cache hit, side by side. The second measures TTFT on the streaming endpoint against total latency on the blocking endpoint, both on a fresh question with the cache empty, and walks the output-guards-deferred trade-off from the previous concept module. The third runs a small `ef_search` sweep against the `scikit_docs` collection and confirms in your own measurements that at a few thousand chunks the curve is measurable, not dramatic — the discipline lesson is that profiling is what tells you whether tuning matters, not faith in defaults. Plan twenty minutes total, weighted toward Exercises 1 and 2 because they exercise the production path the rubric grades.

## Setup

With `make load-data` reporting the scikit-learn corpus is ingested, you should have a live `make serve` ready on port 8080. The `.env` carries `OPENAI_API_KEY` and (on Vocareum) `OPENAI_BASE_URL=https://openai.vocareum.com/v1`. If you are unsure which environment you are on:

```
uv run python -c "from src.config import settings; print(repr(settings.openai_base_url))"
```

You want `''` on direct OpenAI or `'https://openai.vocareum.com/v1'` on Vocareum. Any other value will fail with a 401 on the first embedding call.

You will work in two terminals throughout. The first runs `make serve`; the second is for curl, Python clients, and inspection scripts. Phoenix is in-process — the trace store lives in the FastAPI worker and is cleared when you restart `make serve`. There is no separate Phoenix daemon to bounce.

## Exercise 1 — Profile one query, build the cold-versus-cached table

The rubric §7 evidence target is a per-span latency breakdown that names where the wall-clock budget went. This exercise produces that breakdown twice — once cold and once cached — and the comparison is the artifact you keep.

### What to do

1. Restart the server to start with an empty Phoenix store and a clean cache:

   ```
   # In the make-serve terminal: Ctrl-C, then
   uv run python -c "from src.cache import clear; print('cache removed:', clear())"
   make serve
   ```

   Wait for the `Uvicorn running on http://0.0.0.0:8080` line. The Phoenix store is now empty and the cache collection has zero rows.

2. Fire five identical queries from terminal two. The first will miss the cache and walk the full pipeline; the next four should all hit:

   ```
   for i in 1 2 3 4 5; do
     curl -s -X POST http://localhost:8080/query \
       -H 'Content-Type: application/json' \
       -d '{"question": "How do I tune n_estimators in RandomForestClassifier?"}' \
       | python -c "import sys,json; r=json.load(sys.stdin); print(r.get('cached'), r.get('trace_id','')[:8])"
   done
   ```

   Expected: the first row prints `False`, the next four `True`. If your output looks different, the cache threshold may need a re-check — the semantic-caching module's exercises cover that case.

3. Open `http://localhost:6006` in your browser, navigate to the `scikitdocs` project, and find the five traces from this run. If port 6006 is not reachable in your workspace, use `make show-traces` from a third terminal — that command prints the same per-trace summary as a markdown table. The cold trace will be the one with the long generator span; the cached traces will be much shorter.

4. Click the cold trace. Record the duration of each named span — the classifier call, the cache lookup, the retrieval (embed plus search), and the generator. Then click any cached trace and record the same span list. Some spans will be missing on the cache hit path because the cache short-circuits before the LLM call.

5. Build a two-column table. Magnitudes will vary by region and load; the structure is what you keep:

   ```
   | Span                | Cold (ms)  | Cached (ms) |
   |---------------------|------------|-------------|
   | classifier (LLM)    |    ~XXX    |    ~XXX     |
   | cache lookup        |    ~XX     |    ~XX      |
   | retrieve (embed+search) | ~XX    |    n/a      |
   | generator (LLM)     |   ~XXXX    |    n/a      |
   | TOTAL               |   ~XXXX    |    ~XX      |
   ```

   Compute the speedup as `cold_total / cached_total`. On a paraphrase-heavy workload at the starter's defaults, a cache hit lands roughly an order of magnitude faster than the cold path. Hedge the exact ratio — it depends on how fast the hosted endpoint is on your run and whether the classifier call gets short-circuited by the cache on hits.

### Success criteria

A two-column markdown table with cold and cached per-span latencies for one identical query, plus the speedup ratio. A one-paragraph note naming which span dominates the cold total (typically the generator) and which spans the cache hit eliminates entirely. The interpretation is what makes this exercise more than data-entry. The pattern to articulate is the one the previous concept module framed: cache hits eliminate the LLM call, which is the dominant span, so the speedup compresses the budget from seconds to tens of milliseconds. A miss-rate-versus-hit-rate workload mix is the lever that turns that compression into measured production savings, which is what the cost-monitoring module's cost report and the semantic-caching module's cache exercises already covered.

### Stretch

Fire a query that the embedder will retrieve weakly — for example, ask about a topic the scikit-learn docs do not cover well, like "what is the best PyTorch optimizer?" The retrieval span should fire with low-similarity sources and the response should come back with a low `confidence` field on the `QueryResponse`. The trace makes the low-confidence retrieval visible alongside the happy-path retrieval. Compare the two and reason about whether a confidence threshold gate would be the right place to refuse the query, or whether the generator should be allowed to answer with hedged language. There is no universal answer; the deliberate choice is the point.

A second stretch: open the OpenInference span the `OpenAIInstrumentor` adds under the generator. The attributes `llm.token_count.prompt`, `llm.token_count.completion`, and `llm.token_count.total` are what the cost-monitoring module's cost report aggregates over. Note the relationship between completion-token count and the generator span's duration — longer completions take linearly longer to generate, and that linearity is the load-bearing assumption behind every per-token throughput claim in the previous concept module.

## Exercise 2 — Measure TTFT on the streaming endpoint

The starter runs both `/query` (blocking) and `/query/stream` (SSE). The previous concept module framed the comparison as TTFT-vs-total — same total latency, much faster perceived response. This exercise turns that framing into your own numbers.

### What to do

1. Confirm `make serve` is up, clear the cache one more time so the blocking endpoint gets a fresh cache miss:

   ```
   uv run python -c "from src.cache import clear; print('cache removed:', clear())"
   ```

2. Write a small Python client that times both endpoints on the same question. Save this as `scripts/ttft_compare.py` in your own scratch space — do not modify the starter tree:

   ```python
   import json
   import time
   import urllib.request

   QUESTION = "How do I serialize a fitted scikit-learn Pipeline?"

   def time_blocking() -> dict:
       req = urllib.request.Request(
           "http://localhost:8080/query",
           data=json.dumps({"question": QUESTION}).encode(),
           headers={"Content-Type": "application/json"},
       )
       start = time.perf_counter()
       with urllib.request.urlopen(req) as resp:
           resp.read()
       total_ms = (time.perf_counter() - start) * 1000
       return {"ttft_ms": total_ms, "total_ms": total_ms}

   def time_streaming() -> dict:
       req = urllib.request.Request(
           "http://localhost:8080/query/stream",
           data=json.dumps({"question": QUESTION}).encode(),
           headers={"Content-Type": "application/json"},
       )
       start = time.perf_counter()
       ttft_ms = None
       with urllib.request.urlopen(req) as resp:
           for line in resp:
               if ttft_ms is None:
                   ttft_ms = (time.perf_counter() - start) * 1000
       total_ms = (time.perf_counter() - start) * 1000
       return {"ttft_ms": ttft_ms or total_ms, "total_ms": total_ms}

   print("blocking :", time_blocking())
   print("streaming:", time_streaming())
   ```

3. Run it. Clear the cache between runs so both calls miss:

   ```
   uv run python scripts/ttft_compare.py
   uv run python -c "from src.cache import clear; clear()"
   uv run python scripts/ttft_compare.py
   ```

   Expected shape, hedged on magnitudes:

   ```
   blocking : {'ttft_ms': ~2500, 'total_ms': ~2500}
   streaming: {'ttft_ms': ~500,  'total_ms': ~2500}
   ```

   Blocking TTFT equals blocking total — the client cannot do anything until the whole body lands. Streaming TTFT lands in the few-hundred-millisecond range because the first token arrives as soon as the model starts generating. Total streaming time is comparable to blocking total — the model still has to finish generating, and the network still has to deliver every byte. The streaming route bypasses the cache, so even on a paraphrase-repeat it always pays the full generator cost.

4. Build a two-by-two table:

   ```
   |               | TTFT (ms)  | Total (ms) |
   |---------------|------------|------------|
   | blocking      |   ~XXXX    |   ~XXXX    |
   | streaming     |   ~XXX     |   ~XXXX    |
   ```

5. Open `src/streaming.py` and read the `_pre_stream_guards` docstring and the `query_stream` docstring. That pair names the input-guards-pre-stream-and-output-guards-deferred trade-off explicitly — input guards (prompt-injection plus PII) run before the stream opens when the guardrails module fills the shim, but the hallucination judge and the off-topic check that gate the blocking route are intentionally not applied to the streamed tokens. The starter offers both routes so the trade-off is visible in code. Pick the workload — a low-stakes FAQ versus a high-stakes regulated chatbot — and reason about which route is the right default for each. There is no universal answer; the deliberate choice is the point.

### Success criteria

A two-by-two TTFT-vs-total table with hedged measurements for both routes, on the same question with the same cache state. A one-paragraph interpretation naming the user-perceived difference (the spinner that did not appear) and the engineering cost (the deferred output guards). Honest framing matters here — do not write "streaming is N times faster" because total is comparable. The right framing is "streaming respects the one-second flow-of-thought threshold from Nielsen's 1993 NN/g piece even when total time exceeds it." A second paragraph: name which route you would default your hypothetical docs-FAQ workload to, and why. The starter defaults its primary `/query` route to blocking because the output guards (once the guardrails module lands) need the whole answer to fire; the `/query/stream` route exists as the perceived-latency path for the cases where a single second of spinner is the wrong UX. Most teams pick one as the default and reach for the other when product surface area justifies the operational cost of running both.

### Stretch

Modify your Python client to `print(token, end="", flush=True)` on each `content` field as it arrives. Watch the answer assemble character by character. That is the UX the streaming endpoint enables; the blocking endpoint cannot match it regardless of how fast the generator runs.

A second stretch: fire the same question twice through `/query` (the blocking endpoint) without clearing the cache between calls. The second call returns the cached answer immediately. Now fire the same question twice through `/query/stream`. Both streaming calls always pay the full generator cost because the streaming route bypasses the cache by design. The cache wins on the blocking-route hit path; streaming wins on perceived-latency on the cache-miss path. Each lever solves a different problem.

## Exercise 3 — Sweep `ef_search` against the `scikit_docs` collection

The `scikit_docs` collection at `src/store.py:75-93` uses Chroma's defaults for the three tunable HNSW knobs — `M = 16`, `ef_construction = 100`, `ef_search = 100`. Only `hnsw:space` is set explicitly, because OpenAI embeddings are normalized and L2 silently corrupts ranking against them. The corpus is roughly three to four thousand scikit-learn doc chunks — more than enough for the recall-versus-latency curve to show measurable variance, unlike the capstone's thirty-product workload where the curve is flat below the noise floor.

### What to do

1. Do not edit `src/store.py`. Instead, build separate sandbox collections in a scratch script so the starter's index stays intact. Save this as `scripts/ef_search_sweep.py` in your scratch space:

   ```python
   import time

   import chromadb
   from chromadb.config import Settings as CS

   from src.config import settings
   from src.embedder import embed_query

   client = chromadb.PersistentClient(
       path=settings.chroma_path,
       settings=CS(anonymized_telemetry=False),
   )

   src = client.get_or_create_collection("scikit_docs").get(
       include=["documents", "embeddings", "metadatas"]
   )

   QUERIES = [
       "How do I tune n_estimators in RandomForestClassifier?",
       "What is the default criterion for DecisionTreeRegressor?",
       "How do I use ColumnTransformer with Pipeline?",
       "What does the random_state parameter do?",
       "How do I serialize a fitted estimator?",
   ] * 20  # 100 query repetitions

   def sweep(ef: int) -> dict:
       name = f"scikit_docs_ef_{ef}"
       try:
           client.delete_collection(name)
       except Exception:
           pass
       col = client.get_or_create_collection(
           name,
           metadata={"hnsw:space": "cosine", "hnsw:search_ef": ef},
       )
       col.upsert(
           ids=src["ids"],
           documents=src["documents"],
           embeddings=src["embeddings"],
           metadatas=src["metadatas"],
       )
       embeddings = [embed_query(q) for q in QUERIES]
       start = time.perf_counter()
       for emb in embeddings:
           col.query(query_embeddings=[emb], n_results=5)
       elapsed_ms = (time.perf_counter() - start) * 1000
       return {"ef_search": ef, "mean_ms": elapsed_ms / len(QUERIES)}

   for ef in [10, 50, 200]:
       print(sweep(ef))
   ```

   The script pulls every existing chunk out of the live `scikit_docs` collection, copies them into three parallel collections built with `ef_search` of 10, 50, and 200, then runs the same five queries twenty times against each collection and reports mean per-query latency. The embedding calls are done up front and excluded from the timing so the measurement is the vector search alone.

2. Run it:

   ```
   uv run python scripts/ef_search_sweep.py
   ```

   Expected shape — every row's `mean_ms` should land in the single-digit-to-low-tens-of-milliseconds range, with the three values measurably spread because the corpus is large enough to surface the curve:

   ```
   {'ef_search': 10,  'mean_ms': ~3.2}
   {'ef_search': 50,  'mean_ms': ~5.1}
   {'ef_search': 200, 'mean_ms': ~9.4}
   ```

   Your absolute numbers will differ; the ordering — `ef=200` slower than `ef=10` — is what should be reproducible.

3. Build a recall metric. Hand-pick five queries with a known correct top-1 chunk — pick chunks you have inspected from the live collection (their `doc_id` strings are visible in any `make show-traces` output):

   ```python
   GOLDEN = [
       ("n_estimators default in RandomForestClassifier", "ensemble/forest.rst#L42"),
       ("StandardScaler with_mean argument",              "preprocessing/_data.rst#L15"),
       ("Pipeline named_steps attribute",                 "pipeline.rst#L88"),
       ("ColumnTransformer remainder parameter",          "compose/_column_transformer.rst#L23"),
       ("OneHotEncoder handle_unknown values",            "preprocessing/_encoders.rst#L66"),
   ]
   ```

   For each `ef_search` value, run each golden query and check whether the expected `doc_id` appears in the top-5. Compute `recall@5` as the fraction of golden queries where the expected id was retrieved. At the starter's scale, low `ef_search` values may miss occasionally; higher values should saturate to recall@5 = 1.0. Cite Chroma's collection-configure docs for the parameter semantics.

4. Build the result table:

   ```
   | ef_search | mean latency (ms) | recall@5 |
   |-----------|-------------------|----------|
   |    10     |       ~XX         |   ~0.8   |
   |    50     |       ~XX         |   ~1.0   |
   |   200     |       ~XX         |   ~1.0   |
   ```

5. Clean up the sandbox collections so the next exercise's vector DB is unchanged:

   ```
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

### Success criteria

A three-row table — `ef_search` versus mean latency versus recall@5. A one-paragraph interpretation. The expected interpretation: at a few thousand chunks the curve has measurable variance; latency rises with `ef_search` and recall tends to saturate by `ef_search = 50` or so. The right operational read is that the LLM call still dominates the wall-clock budget on this workload, so tuning effort would still be better spent on the cache hit rate (the semantic-caching module) or the model choice (the cost-monitoring module's cost report). The same script and the same discipline apply at ten-thousand and ten-million chunk scale, where the curve sharpens and the tuning earns its keep.

A second paragraph: name which knob you would tune first on a hypothetical ten-million-row workload, given the cost structure you already measured in Exercise 1. The right answer is rarely vector search — even at large scale the LLM call usually still dominates, and the cache and the model-choice levers compound multiplicatively against it. Pinecone's HNSW article walks the recall-versus-latency curve on the Sift1M benchmark; that is the shape you would replicate against your own workload before committing to an `ef_search` value in production.

## Common Pitfalls

- **Forgetting `stream_options={"include_usage": True}`.** Without that flag in `src/streaming.py`, the streaming response carries no token counts and the cost computation silently reports zero. The cost-monitoring module's dashboard will under-report streaming traffic. If you copy this pattern outside the starter, the option is the line you must not drop.
- **Comparing TTFT-stream to Total-blocking and claiming "streaming is N times faster."** That comparison mixes two different metrics. Total time is comparable across both endpoints; streaming wins on TTFT and on the user-perceived experience. The previous concept module framed this as the correct mental model and the wrong mental model side by side.
- **Cache contamination in the sweep.** The blocking route uses the cache; the streaming route does not. Between sweep iterations on the blocking endpoint, clear the cache or you will measure cache-hit latency instead of vector search latency. Exercise 1 demonstrates the cache-hit path on purpose; Exercise 3 wants the opposite — the vector search measured in isolation, with the cache out of the loop entirely. The sandbox-collection approach in Exercise 3 sidesteps the cache because it calls Chroma directly without going through the gateway.
- **Editing `src/store.py` to change HNSW params, then re-running `make load-data`.** Chroma builds the HNSW graph once at insert time. Changing `M` or `ef_construction` on an already-built collection has no effect — you have to drop the collection first (`rm -rf data/chroma/`) and rebuild. The sandbox-collection pattern in Exercise 3 avoids the trap by creating fresh parallel collections with the parameters baked in at creation.
- **Phoenix not running when you expect spans.** If you set `TRACING_BACKEND=none` in `.env` or if Phoenix failed to launch (check `make serve` startup logs for the tracing-init line), the `/query` response will still carry `trace_id=""` and the UI at port 6006 will be empty. `make show-traces` will print "no traces found." Restart with the default backend and the spans return.
- **Reading the Phoenix UI without remembering it is in-process.** The Phoenix store lives in the FastAPI worker. Restarting `make serve` clears it. If you ran a query yesterday and expect to see it today, you will not — the trace store is not persisted across restarts in the embedded configuration. The rubric §7 evidence path expects you to capture the trace export from `make show-traces` or screenshot the UI within the same session as the queries that produced the spans.
- **Measuring TTFT with curl's default buffering.** Without `-N`, curl buffers the entire response body before printing, which means the wall-clock time you see is total, not TTFT. The Python client in Exercise 2 uses `urllib.request.urlopen` which is line-buffered by default; the per-iteration `for line in resp` loop is what makes the TTFT measurement meaningful. If you adapt this to a different HTTP library, confirm that it does not buffer the response body before yielding the first chunk.
- **Expecting `/query/stream` to hit the cache on a paraphrase.** The streaming route bypasses the cache by design — see the implementation note in Demo Part 2. If your streaming TTFT numbers look implausibly identical run-over-run on the same question, that is the expected behavior. Cache wins are measured on `/query`, not on `/query/stream`.

What you have at the end. A Phoenix-rendered latency breakdown for one query, cold versus cached. A TTFT-versus-total comparison between the streaming and blocking endpoints with hedged magnitudes. An `ef_search` sweep that demonstrates in your own numbers how the curve behaves at a few thousand chunks and where it would start to matter at production scale. Three artifacts that together cover the profile-streaming-tune triangle the previous concept module framed, on the exact code paths the starter ships.

Two forward references close the loop. The vLLM concept module picks up the self-hosted inference path — when the API-hosted model is the bottleneck and the volume justifies the operational overhead, batching and KV-cache tuning open up the inference-side levers that hosted endpoints do not expose. The course-review module is the bridging unit that ties together the threads from every prior module; the trace-reading discipline and the cache-as-primary-savings-lever framing from this module are two of the threads it pulls forward into the capstone wrap-up.

One closing observation about the discipline this exercise is building. The hardest part of latency engineering is not the measurement — Phoenix is in-process, the spans are auto-instrumented, the curl commands are five lines each. The hard part is resisting the urge to tune the thing you can tune instead of the thing that matters. The HNSW knobs in `src/store.py` are tunable; the LLM call's TTFT mostly is not (unless you swap models or self-host). The exercises above are designed so that you measure both surfaces and arrive at the right ratio of effort — cache and model choice first, vector search later, embedding model last. That ratio is the operational habit the previous concept module named and this module demonstrated on the starter. Carry it forward.
