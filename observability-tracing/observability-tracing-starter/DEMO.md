> A walkthrough of the codebase you'll work with. See INSTRUCTIONS.md for the exercise tasks.

# Demo: Wire Phoenix into the ScikitDocs Pipeline and Read One Trace

Traces, spans, attributes, the OpenTelemetry and OpenInference standards on top — this demo connects that vocabulary to running code. You will read `src/tracing.py` end to end, fire one traced `run_pipeline` call against the ScikitDocs corpus, open the Phoenix UI at `localhost:6006`, and read the six-span tree it produced. Then you will export the same data as markdown with `make seed-traces` — the rubric §7 evidence path for environments where the UI port is not reachable.

## Why Arize Phoenix

The starter uses **Arize Phoenix** as its tracing backend. The reason is operational: Phoenix runs in-process inside any Python entry point with no SaaS account, no Docker daemon, and no API key — which fits the Udacity Workspace constraints (4 GB RAM, 2 vCPU, no AWS). Phoenix's community version is Apache 2.0, exposes its UI at `localhost:6006`, and stores traces in a local SQLite database under `data/phoenix/` when persistence is configured.

The patterns transfer. The trace UI, the span tree, the attribute schema, and the diagnostic workflows look the same in LangSmith, Langfuse, and Datadog LLM Observability. The choice here is deployment topology, not concepts.

## Setup

You should already have the ScikitDocs starter loaded — `make setup` ran, `.env` is populated with `OPENAI_API_KEY` (plus `OPENAI_BASE_URL=https://openai.vocareum.com/v1` if you are on Vocareum), and `make load-data` printed `Wrote N chunks`. If not, run those first; without the Chroma store populated, the `traced_pipeline` calls below return empty source lists.

The starter ships a FastAPI gateway at `src/gateway/`, but this module focuses on the tracing surface itself rather than the HTTP entry point. Instead of `make serve` + a curl loop, you boot tracing and fire queries through direct Python calls. The wiring inside `init_tracing` is identical to what the gateway lifespan calls; the difference is just who calls it.

## Walkthrough 1 — `src/tracing.py`, end to end

Open the file side by side with `src/pipeline.py`. They are small and worth reading in full.

The settings the tracer reads from live in `src/config.py:36-41`:

```python
tracing_backend: Literal["phoenix", "none"] = "phoenix"
phoenix_embedded: bool = True
phoenix_host: str = "0.0.0.0"
phoenix_port: int = constants.PHOENIX_PORT          # 6006
phoenix_working_dir: str = "data/phoenix"
phoenix_project_name: str = constants.PHOENIX_PROJECT_NAME  # "scikitdocs"
```

`tracing_backend="none"` is the kill-switch the test suite uses to skip Phoenix entirely — useful for CI and any environment where you want to test the pipeline without launching the embedded UI. `phoenix_embedded=True` is the default we ship; setting it to `False` would point Phoenix at an external collector instead, which is the path you take when you outgrow the single-process model. The two `constants.*` references hold the locked values from `src/constants.py` — port 6006 and the project name `scikitdocs`.

`src/tracing.py:53-90` is `init_tracing`, the function that wires everything. Three calls are doing the work:

```python
_phoenix_session = px.launch_app()
...
_tracer_provider = register(
    project_name=settings.phoenix_project_name,
    endpoint=endpoint,
    verbose=False,
)
OpenAIInstrumentor().instrument(tracer_provider=_tracer_provider)
```

`px.launch_app()` starts the in-process Phoenix UI bound to the host and port from settings. `phoenix.otel.register(...)` registers an OpenTelemetry `TracerProvider` configured to export spans to that Phoenix instance over OTLP, under the project name `scikitdocs`. `OpenAIInstrumentor().instrument(...)` patches the OpenAI SDK so every chat-completions and embeddings call emits an OpenInference-shaped span with the model name, input messages, output messages, and token counts as attributes. You wrote zero instrumentation code in `src/embedder.py` or `src/generator.py` — the instrumentor does the work at the SDK seam.

The custom root + child spans live in `traced_pipeline` further down the same file, at `src/tracing.py:120-205`. The function imports `embed_query`, `query`, `render_system_prompt`, and `generate` directly (rather than calling `run_pipeline`) so each stage gets its own named span. The resulting hierarchy:

```
rag_query           (root, the request)
  ├── retrieve
  │   ├── embed     (auto-child: OpenAI Embedding span)
  │   └── search    (Chroma cosine top-k)
  ├── augment       (Jinja system-prompt render)
  └── generate      (auto-child: OpenAI ChatCompletion span)
```

The duplication with `src/pipeline.py` is deliberate. The plain RAG composition in `pipeline.py` stays free of any opentelemetry import; the tracing wrapper here is where the instrumentation surface lives. Each named span carries the attributes the eval, cost, and latency layers read from, plus a 32-character hex `trace_id` injected back into `QueryResponse` via `model_copy(update={"trace_id": ...})`.

## Walkthrough 2 — fire one traced query, read the span tree

Open a Python session. `python -i` keeps Phoenix alive after the call so you can click around the UI:

```
uv run python -i -c "
from src.tracing import init_tracing, traced_pipeline
init_tracing()
r = traced_pipeline('What is the default value of n_estimators in RandomForestClassifier?')
print('trace_id=', r.trace_id)
print('model=', r.model, 'confidence=', round(r.confidence, 3), 'tokens=', r.tokens.total)
"
```

A representative response shape (numbers will vary):

```
trace_id= a99da61c30f89d026127d7dd3d3873c8
model= gpt-4o confidence= 0.529 tokens= 1518
```

The `trace_id` is the 32-hex string `init_tracing`'s OTel registration produced and the wrapper at `tracing.py:198-202` copied back onto the response. If `localhost:6006` is reachable from your browser, navigate to the `scikitdocs` project, find the newest trace, and click in. You should see the six-span tree from Walkthrough 1: a root `rag_query`, a `retrieve` parent with `embed` and `search` children, an `augment` sibling, and a `generate` sibling. The `OpenAIInstrumentor` adds its own auto-spans inside `embed` (a `ChatCompletion`-named Embedding span — Phoenix labels it by the OpenAI SDK call name) and inside `generate` (the actual chat-completions call, with input messages and output message visible as attributes).

Click the root `rag_query` span and read the attributes panel. You will see `rag.latency_ms`, `rag.confidence`, `rag.cost_usd`, `rag.top_k`, `rag.model`, and a `rag.sources` field stringifying the doc IDs and similarity scores. Those are the fields the wrapper at `tracing.py:188-195` attached. The `input.value` field carries the question; `output.value` carries the answer. The `retrieve` span carries `rag.top_k` and `rag.sources.count`, and its `search` child carries `rag.sources.top_score`. The `generate` child carries `llm.token_count.{prompt,completion,total}` and `rag.cost_usd` for the per-call cost (populated by `src/pricing.py`).

That is the full trace surface for one request. Six named spans, every attribute set explicitly in code, plus the two auto-instrumented child spans the OpenAIInstrumentor contributes for free.

A small filter-and-aggregate walk through the same UI: in the Phoenix project view, sort traces by total duration descending — the slowest single trace is the entry point for any latency investigation. Sort by `llm.token_count.total` to surface the most expensive request, which is the entry point for cost analysis. Filter on `status == ERROR` to surface failures; the wrapper at `tracing.py:177-180` records the exception on the root span when any stage raises, so any pipeline error lands as a queryable trace rather than a silent log line.

## Walkthrough 3 — `make seed-traces` as the §7 backstop

If `localhost:6006` is blocked in your workspace, you still need an evidence path for the rubric. Run:

```
make seed-traces
```

That invokes `scripts/seed_traces.py`, which calls `init_tracing()`, fires a hand-curated 5-question pack (in-domain factual, version-sensitive, conceptual, off-topic, multi-API comparison) through `traced_pipeline`, waits a few seconds for the OTel exporter to flush, then renders a per-trace markdown table — trace ID, question, model, latency, prompt and completion token counts, slowest child span and its duration. It writes the result to `data/trace_evidence.md` and prints a "slowest step across N traces" one-liner at the foot. Sample row:

```
| 1 | `875871fb` | What is the default value of `n_estimators` ... | gpt-4o | 4833.0 | 1478 | 47 | generate | 2649.3 |
```

The `Slowest child` column is the diagnostic hook. On most runs against `gpt-4o` you will see `generate` as the slowest child, taking the majority of the request's wall-clock time — embedding is an order of magnitude cheaper than generation. When something else dominates the slot — `retrieve` or `augment` — that is the signal that retrieval got slow or the prompt grew, not the LLM.

`make show-traces` is the sibling command for ad-hoc reads when a long-running process is already up (`make serve` is that process). When you are not running the gateway, prefer `make seed-traces`: the fire-and-export happens in the same Python process, so the embedded Phoenix store does not vanish between firing and reading.

## Wrap-up

Three calls wire Phoenix in: `px.launch_app()`, `phoenix.otel.register(...)`, and `OpenAIInstrumentor().instrument(...)`. One wrapper — `traced_pipeline` — composes the pipeline with six named spans matching the request → retrieve → embed → search → augment → generate hierarchy. One Python invocation produces a complete trace, viewable in the UI at `localhost:6006` or as markdown via `make seed-traces` at `data/trace_evidence.md`. The exercises take this further: open one trace in the UI, export the rubric §7 evidence file, and edit `src/tracing.py` to add a custom span attribute the rest of the stack reads.

One operational note before you move on. The trace evidence file is regeneratable — `make seed-traces` overwrites it on every run, so checking the file in is not a one-shot ritual. It is the simplest evidence path a reviewer can verify without running anything, and it stays useful as the corpus, models, and retrieval parameters evolve. The eval, cost, and latency layers each read attributes off the same span tree; the wiring you watched come together in this demo is the data backbone for everything operational that follows.

---


