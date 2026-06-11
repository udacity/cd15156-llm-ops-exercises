> A walkthrough of the codebase you'll work with. See INSTRUCTIONS.md for the exercise tasks.

# Demo — Profile a Query, Stream the Answer, See Where the HNSW Knobs Live

The previous concept module named the principle — profile before you optimize, time-to-first-token and total latency are different metrics, vector-search tuning is small at small scale. This demo brings all three down to the ScikitDocs starter. You will fire one query through the live gateway, open the Phoenix trace it produced, walk the streaming endpoint that ships with the starter, and look at the place in `src/store.py` where the HNSW knobs would go if the corpus grew to ten million rows. The point is to read what is already there with the right vocabulary.

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

Output guards do not run on the blocking path the starter ships today. A hallucination judge and an off-topic check that fire after generation are the guardrails surface — those scanners run against the same gateway and add latency you can measure. For this demo, the blocking `/query` route runs classifier → cache → retrieve → generate and returns. That is the surface Exercise 1 profiles.

Stack the numbers. On a cache miss, the classifier and the generator dominate the wall-clock latency, with the rest distributed across cache lookup, retrieval, and overhead. That distribution is what should drive optimization effort. The cache is the biggest single win on paraphrase-heavy traffic because a hit eliminates the classifier, the retrieval, and the generator spans in one move. The previous concept module named this; Phoenix shows it.

## Part 2 — Streaming the answer with `/query/stream`

The starter ships two query endpoints. The blocking `/query` runs the full pipeline and returns a single JSON body — that is what Part 1 traced. The streaming `/query/stream` is mounted from a separate file at `src/streaming.py` and emits Server-Sent Events token by token. Both routes are wired in by `create_app` at `src/gateway/app.py:42-51`, which calls `application.include_router(streaming_router)` alongside the blocking router and the cost-monitoring module's dashboard.

Read `src/streaming.py` first. The function that does the work is `stream_completion` at the middle of the file. The signature returns `Iterator[str | tuple[str, TokenUsage, float]]` — each iteration yields a token as a string, and the final iteration yields a three-tuple of the assembled answer, the token usage, and the dollar cost. The key API choice is the `stream_options` argument:

```python
stream_options={"include_usage": True},
```

Without that flag, the OpenAI streaming response does not include the `usage` field on its final chunk. The cost computation that follows — `compute_cost(model, usage)` — has nothing to work from, and the cost dashboard reports zero. This is the gotcha the OpenAI cookbook calls out and the starter codes around explicitly. If you copy this pattern to another project, the include-usage option is the line you must not drop.

Now walk the SSE route. `query_stream` at the bottom of `src/streaming.py` wires `_stream` into FastAPI's `StreamingResponse` with `media_type="text/event-stream"`. The input-guards seam is `_pre_stream_guards(question)` — a thin function that today returns `(question, None)`. The seam is where input guards would run: the LLM Guard prompt-injection scanner and the Presidio PII redactor before any token is emitted; a prompt-injection match short-circuits to `_blocked_stream` and the response carries exactly one `done` SSE event with `blocked_by` set, and PII in the question is rewritten in place before retrieval and generation. Output guards over a streamed response are intentionally deferred. The hallucination judge and off-topic check both need the whole answer to make a decision, and applying them on a half-finished stream is the streaming-guards composition problem the concept module described. On this workload, the cleaner answer is to either run the full guards on the blocking route or defer them as a follow-up on the streaming route. Both choices live in the codebase so the trade-off is visible.

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

The honest framing. At the starter's three-to-four-thousand chunks the curve is measurable but not dramatic — single-digit to low-tens of milliseconds across the typical sweep range. The variance is visible here because more chunks mean more navigation work for the algorithm. At a few million the tuning becomes operationally meaningful and the recall-versus-latency trade becomes a real engineering decision. The discipline transfers regardless of scale: profile before you tune, and tune the knob whose curve sits above the noise floor on your specific workload. The exercise that follows lets you run a sweep on the starter's corpus and see for yourself.

One more lever to name. The other latency lever at the model layer — running inference yourself on a GPU — is self-hosting. When the API-hosted model is the bottleneck and you have sustained volume to justify the operational load, self-hosting opens up batching and KV-cache tuning that hosted endpoints do not expose. Most teams do not reach that point. Knowing it exists is the takeaway.

---

