# Module 09 — Implement LLM Call Tracing with Phoenix

## Setup

This starter is the ScikitDocs RAG app with the prompt loader, vector database, RAG pipeline, tracing, evaluation, cost monitoring, semantic caching, gateway, guardrails, A/B testing, RAGOps watcher, and latency optimization already wired. In this module you will (1) read `src/tracing.py` end to end and fire one traced query against the corpus, (2) export the rubric §7 evidence file via `make seed-traces`, and (3) add one custom span attribute (`rag.retrieve.top_score`) to the `retrieve` span. Run `make setup && make load-data` to bring up the corpus, then follow the demo walkthrough and the exercise tasks below.


# Module 09 — Demo: Wire Phoenix into the ScikitDocs Pipeline and Read One Trace

Module 08 named the vocabulary — traces, spans, attributes, the OpenTelemetry and OpenInference standards on top. This demo connects that vocabulary to running code. You will read `src/tracing.py` end to end, fire one traced `run_pipeline` call against the ScikitDocs corpus, open the Phoenix UI at `localhost:6006`, and read the six-span tree it produced. Then you will export the same data as markdown with `make seed-traces` — the rubric §7 evidence path for environments where the UI port is not reachable.

## One CSV-vs-capstone deviation to name up front

The submitted module dictionary calls this module "Implement LLM Call Tracing with LangSmith." The starter uses **Arize Phoenix**, not LangSmith, per the locked decision in `docs/2026-04-29-udacity-deliverable-deltas.md` §7.4. The reason is operational: Phoenix runs in-process inside any Python entry point with no SaaS account, no Docker daemon, and no API key — which fits the Udacity Workspace constraints (4 GB RAM, 2 vCPU, no AWS). Phoenix's community version is Apache 2.0, exposes its UI at `localhost:6006`, and stores traces in a local SQLite database under `data/phoenix/` when persistence is configured.

The patterns transfer. Module 08 covered the ecosystem comparison; the trace UI, the span tree, the attribute schema, and the diagnostic workflows look the same in LangSmith, Langfuse, and Datadog LLM Observability. The choice here is deployment topology, not concepts.

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

The custom root + child spans live in `traced_pipeline` further down the same file, at `src/tracing.py:120-205`. The function imports `embed_query`, `query`, `render_system_prompt`, and `generate` directly (rather than calling `run_pipeline`) so each stage gets its own named span. The resulting hierarchy matches the vocabulary Module 08 named in the abstract:

```
rag_query           (root, the request)
  ├── retrieve
  │   ├── embed     (auto-child: OpenAI Embedding span)
  │   └── search    (Chroma cosine top-k)
  ├── augment       (Jinja system-prompt render)
  └── generate      (auto-child: OpenAI ChatCompletion span)
```

The duplication with `src/pipeline.py` is deliberate. The plain RAG composition in `pipeline.py` stays free of any opentelemetry import; the tracing wrapper here is where the instrumentation surface lives. Each named span carries the attributes the eval, cost, and latency layers will read from later, plus a 32-character hex `trace_id` injected back into `QueryResponse` via `model_copy(update={"trace_id": ...})`.

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

Click the root `rag_query` span and read the attributes panel. You will see `rag.latency_ms`, `rag.confidence`, `rag.cost_usd`, `rag.top_k`, `rag.model`, and a `rag.sources` field stringifying the doc IDs and similarity scores. Those are the fields the wrapper at `tracing.py:188-195` attached. The `input.value` field carries the question; `output.value` carries the answer. The `retrieve` span carries `rag.sources.count` and `rag.sources.top_score`. The `generate` child carries `llm.token_count.{prompt,completion,total}` and `rag.cost_usd` for the per-call cost (populated by `src/pricing.py`).

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

Three calls wire Phoenix in: `px.launch_app()`, `phoenix.otel.register(...)`, and `OpenAIInstrumentor().instrument(...)`. One wrapper — `traced_pipeline` — composes the pipeline with six named spans matching the request → retrieve → embed → search → augment → generate hierarchy from Module 08. One Python invocation produces a complete trace, viewable in the UI at `localhost:6006` or as markdown via `make seed-traces` at `data/trace_evidence.md`. The exercises take this further: open one trace in the UI, export the rubric §7 evidence file, and edit `src/tracing.py` to add a custom span attribute the rest of the stack reads.

One operational note before you move on. The trace evidence file is regeneratable — `make seed-traces` overwrites it on every run, so checking the file in is not a one-shot ritual. It is the simplest evidence path a reviewer can verify without running anything, and it stays useful as the corpus, models, and retrieval parameters evolve across the rest of the course. Modules 11, 13, and 26 will each read attributes off the same span tree; the wiring you watched come together in this demo is the data backbone for everything operational that follows.

---


# Module 09 — Exercise: Read One Trace, Export the §7 Evidence File, Add a Custom Span Attribute

The demo wired Phoenix into the ScikitDocs pipeline and walked one traced query through the UI. These exercises make the workflow producible. Three exercises, each ending in a small artifact you can paste into a writeup or check into the repo: a one-paragraph description of a trace you opened yourself, the rubric §7 evidence file at `data/trace_evidence.md`, and a four-line diff that adds a new attribute the downstream modules will read. Plan for twenty minutes total, weighted slightly toward exercise 2 because the evidence-file workflow is the canonical path for environments without browser access to `localhost:6006`.

## Setup

Same setup as the demo. ScikitDocs starter cloned, `make setup` ran, `.env` populated with `OPENAI_API_KEY` (plus `OPENAI_BASE_URL=https://openai.vocareum.com/v1` if you are on Vocareum), `make load-data` printed `Wrote N chunks`. Each exercise fires queries through direct Python rather than `make serve` + curl — the tracing surface is identical either way, but a short-lived Python process keeps the example focused on the spans rather than the HTTP layer.

If `localhost:6006` is reachable from your browser, open it now. If it is not, you will lean on `make seed-traces` and the `data/trace_evidence.md` artifact — that is the rubric §7 backstop, and exercise 2 is exactly the workflow for environments without UI access.

## Exercise 1 — Read one trace in the Phoenix UI

The goal of this exercise is to develop the muscle of looking at a trace before trusting any aggregate dashboard. You will fire three traced queries, find one of them in the Phoenix UI, expand the span tree, and write up what you see.

### What to do

1. Start an interactive Python session so Phoenix stays alive while you click around. Use `python -i` so the process does not exit after the calls:

   ```
   uv run python -i -c "
   from src.tracing import init_tracing, traced_pipeline
   init_tracing()
   for q in [
       'What is the default value of n_estimators in RandomForestClassifier?',
       'Explain how StandardScaler works.',
       'Compare RandomForestClassifier and GradientBoostingClassifier for tabular data.',
   ]:
       r = traced_pipeline(q)
       print(r.trace_id, r.model, r.tokens.total)
   "
   ```

   Three lines print, each with a 32-character hex `trace_id`. Keep them handy — you will use one of them to find a specific trace in the UI.

2. Open `http://localhost:6006/` in your browser. Pick the `scikitdocs` project (the project name comes from `constants.PHOENIX_PROJECT_NAME`). You should see three traces in the list, newest first.

3. Click into one of them. The Gantt-chart view in the right pane shows the span tree. You are looking at the same six-span hierarchy the demo named: a root `rag_query`, a `retrieve` parent with `embed` and `search` children, an `augment` sibling, a `generate` sibling, and the OpenAIInstrumentor's auto-children inside `embed` and `generate`.

4. Click the root `rag_query` span. Read the attributes panel on the right. Identify and write down five attributes — preferred picks: `rag.latency_ms`, `rag.confidence`, `rag.cost_usd`, `rag.model`, and `rag.sources`. For each, write one sentence about what it tells you operationally. Example: *`rag.confidence = 0.529` — mean similarity score across the retrieved chunks, in [0, 1]; below the `CONFIDENCE_THRESHOLD = 0.7` constant, so the answer is one a production system would surface a hedge for.*

5. Write a one-paragraph description of the span tree: which span dominated total duration, what does that imply about the bottleneck, and what auto-spans the OpenAIInstrumentor contributed inside `embed` and `generate`.

### Acceptance criterion

Three trace IDs printed in your terminal, plus a one-paragraph description of the span tree for one of them, plus a five-attribute glossary. The writeup should be specific enough that a teammate could open the same trace in your Phoenix UI and confirm each claim against the attributes panel.

### Hints

<details>
<summary>If `localhost:6006` does not respond in your browser</summary>

Two common causes. First, the Phoenix UI mounts when `init_tracing()` runs — if your Python session exited before you tried the browser, the embedded server is gone. Re-run with `python -i` so the session stays alive. Second, some Udacity Workspace configurations block direct browser access to port 6006; `make serve-proxy` is the gateway-side workaround (sets `PHOENIX_HOST_ROOT_PATH=/proxy/6006` so the UI's HTML and JS bundles fetch assets from the right prefix). If neither path works, the documented backstop is `make seed-traces` → `data/trace_evidence.md`, which exercise 2 covers in full.
</details>

<details>
<summary>If the `trace_id` field is `None` in the printed output</summary>

`init_tracing()` did not run, or `TRACING_BACKEND=none` is set in your `.env`. The wrapper still produces a `QueryResponse` either way — the OTel API returns a no-op tracer when no provider has been registered, the spans go nowhere, and `format(ctx.trace_id, "032x")` returns a string of zeros that the wrapper substitutes back to `None`. Check the printed output of `init_tracing()` for the Phoenix banner ("🌍 To view the Phoenix app in your browser, visit http://localhost:6006/"). If it is missing, your settings have the backend off; either remove the `.env` override or re-export `TRACING_BACKEND=phoenix` in the shell.
</details>

<details>
<summary>If you see traces in the UI that did not come from your three queries</summary>

Phoenix captures every OpenAI SDK call as its own trace by default — the embedding requests for the retrieval step show up alongside the `rag_query`-rooted ones. The filter to use is `name == "rag_query"`; that scopes the list to the request-level traces the exercise asks you to read. In the JSON export from `scripts/show_traces.py --json`, the same filter happens automatically via the `summarize_traces` groupby logic.
</details>

## Exercise 2 — Export the rubric §7 evidence file

The rubric for the capstone project specifies that §7 evidence is either a screenshot of the Phoenix UI showing a representative trace with spans labeled, or the markdown output of `make seed-traces` for environments where port 6006 is not reachable. This exercise produces the markdown path end to end and is the canonical workflow for any environment without browser access to `localhost:6006`.

### What to do

1. Run the seed script:

   ```
   make seed-traces
   ```

   The script imports `init_tracing`, calls it once, fires the five-question pack against the running starter, waits three seconds for the OTel exporter to flush, and writes the result to `data/trace_evidence.md`. The five questions are picked to exercise different paths: a short fact (`n_estimators` default), a version-sensitive value (`LogisticRegression` solver in 1.5), a multi-paragraph explainer (`StandardScaler`), an off-topic refusal (weather), and a multi-API comparison (`RandomForest` vs `GradientBoosting`).

2. Inspect the file:

   ```
   cat data/trace_evidence.md
   ```

   The top is a five-row table — one row per trace — with columns for trace ID, question, model, latency in milliseconds, prompt token count, completion token count, slowest child span, and slowest-span duration. The bottom is the one-line "slowest step across N traces" summary that the rubric reviewer is looking for.

3. Pick two trace rows that illustrate distinct diagnostic patterns. Conventional picks: the slowest single trace (the row with the highest `Latency (ms)` value) and one other interesting trace — typically the largest prompt token count (a multi-API comparison or an explainer with many retrieved chunks) or the off-topic refusal (which routes through the same six spans but produces a short completion).

4. Write two short paragraphs of commentary, one per chosen trace. For the slow trace: which span dominated, and what does that mean operationally — was generation slow because the prompt was long, or because the model produced a long answer? For the second trace: name the diagnostic pattern you picked it to illustrate, and what it tells you about the request shape.

5. Check `data/trace_evidence.md` in alongside your commentary so the rubric reviewer can verify the table independently of the prose.

### Acceptance criterion

A `data/trace_evidence.md` file containing the five-row table plus the slowest-step one-liner, and a two-paragraph commentary block quoting two specific rows. The output should be paste-ready for the §7 section of a project writeup. If you also have a screenshot of the Phoenix UI showing the same data, include it — the rubric accepts either evidence path, and a screenshot plus the markdown export together is the most defensible submission.

### Hints

<details>
<summary>If `make seed-traces` prints "Could not connect to Phoenix"</summary>

The script handles `init_tracing` for you, so this error usually means a fresh Phoenix bind failed — most often because another process is already holding port 6006 (an interactive `python -i` session from exercise 1 that you forgot to exit, or a stale `make seed-traces` from a previous run). Find and stop the holder, or change `phoenix_port` in `.env` to a free port. The seed script reads `settings.phoenix_port` so an override in `.env` flows through automatically.
</details>

<details>
<summary>If `data/trace_evidence.md` shows fewer than five rows</summary>

Either the OTel exporter has not finished flushing yet (the script waits three seconds by default; pass `--flush-wait-seconds 5` for a slower environment) or one of the five queries raised an exception that the wrapper recorded as an ERROR-status trace — those rows still show up but with a truncated attribute set. Inspect the script's stderr output for the per-query trace ID lines; if any printed `trace=—`, that call failed inside `traced_pipeline` and the trace ID was never set.
</details>

<details>
<summary>If you need a different five-question pack</summary>

The questions live at the top of `scripts/seed_traces.py` as a module-level `_QUESTIONS` list. Replace them with your own; the script imposes no semantic constraints beyond expecting strings. The default mix is chosen to span the four shapes the rubric reviewer cares about (factual, comparative, conceptual, off-topic), so swapping in your own questions is fine as long as the same diagnostic patterns show up.
</details>

## Exercise 3 — Add a custom span attribute

The tracing layer is editable. The attributes you can read in the Phoenix UI are exactly the ones the wrapper at `src/tracing.py:120-205` set explicitly via `span.set_attribute(...)`. This exercise asks you to add one new attribute that surfaces a diagnostic the existing schema does not — the top similarity score returned by retrieval — and verify it appears in the UI and in the markdown export.

### What to do

1. Open `src/tracing.py`. Inside the `with tracer.start_as_current_span("retrieve") as retrieve_span:` block, find the line where `retrieve_span.set_attribute("rag.sources.count", len(sources))` is set on the outer retrieve span (just inside the search block's close). Add one line above it on the retrieve span:

   ```python
   retrieve_span.set_attribute(
       "rag.retrieve.top_score",
       max(s.similarity_score for s in sources) if sources else 0.0,
   )
   ```

   That puts the single largest cosine similarity from the retrieved chunks on the `retrieve` span itself, alongside `rag.sources.count`. The downstream picture: the RAGAS evaluation layer reads this as the retrieval-quality signal for context-precision; the latency layer reads it to filter out low-quality retrievals when sorting by latency tail. The attribute name follows the same `rag.<stage>.<field>` namespacing the rest of the schema uses.

2. Re-fire one query, in the same process so the new instrumentation runs:

   ```
   uv run python -c "
   from src.tracing import init_tracing, traced_pipeline
   init_tracing()
   r = traced_pipeline('What does StandardScaler do?')
   print(r.trace_id)
   "
   ```

3. Verify the new attribute landed. Two paths:

   - **UI path** — open `localhost:6006/`, find the new trace, click the `retrieve` span, scroll the attributes panel until you see `rag.retrieve.top_score`. Confirm the value is in `[0, 1]`. For a well-retrieved scikit-learn API question, expect something between 0.5 and 0.75.
   - **Export path** — run `make seed-traces` (which fires five fresh queries plus this attribute) and open `data/trace_evidence.md`. The summary table does not surface custom retrieve-span attributes by default — that is a render choice in `summarize_traces`, not a data loss — but the JSON export does: `uv run python scripts/show_traces.py --json | head -50` shows the raw span tree. Confirm `rag.retrieve.top_score` appears on the `retrieve` span entries.

4. Write a four-line diff snippet of the change for your writeup. (The line you added is the entire diff — the surrounding context is just for orientation.) Below it, paste the JSON snippet or screenshot that proves the attribute landed in Phoenix.

### Acceptance criterion

A four-line diff snippet showing the `set_attribute("rag.retrieve.top_score", ...)` call you added, plus evidence (a JSON excerpt from `--json` output, a UI screenshot, or both) that the attribute is now visible on every new traced query. The new attribute should not break any existing test — `uv run pytest tests/` should still pass.

### Hints

<details>
<summary>If `rag.retrieve.top_score` does not appear in the UI</summary>

Two common causes. First, the edit lives in a Python file that was already imported in a long-running session — restart the session so `src.tracing` is freshly imported. Second, the attribute is set on a span that already closed before you read it — confirm your edit is inside the `with tracer.start_as_current_span("retrieve"):` block (not after it). The `with` statement closes the span on block exit; attributes set after the close go nowhere because OTel batches by span lifetime.
</details>

<details>
<summary>If you also want the attribute on the root `rag_query` span</summary>

That is a reasonable second edit: a top-score signal at the root span makes it queryable from the trace list view without expanding the tree. Add the same line, replacing `retrieve_span` with `span` (the outer `with` is named `span` for the root), inside the post-pipeline block where the other root attributes are set. The downstream cost is a small attribute-duplication; the trade-off is filterability. The cost dashboard and the RAGAS eval suite both read root-level attributes first, so this is the conventional placement when the same value is useful at multiple levels.
</details>

## Common pitfalls

A few traps that catch most learners on this module:

- **Phoenix UI not loading at `localhost:6006`** — Phoenix only runs while a Python process that called `init_tracing()` is alive (either `make serve` or an interactive session). If the process has exited, the UI is gone. `python -i` keeps the session alive; `make seed-traces` is the in-process fire-and-export pattern that does not need the UI at all.
- **`init_tracing()` was never called, so `trace_id` is `None`** — most common cause is `TRACING_BACKEND=none` in `.env` (kill-switch the test suite uses), or `init_tracing()` was simply missing from the entry point. A `trace_id` field that is `None` in the returned `QueryResponse` is the most reliable signal that the tracer never registered.
- **`make show-traces` empty output** — Phoenix's embedded store is per-process. If you fired queries in a Python session that has since exited, the spans are gone. `make seed-traces` avoids the problem by firing and exporting in the same process; `make show-traces` is the right tool only when a long-running process is up (`make serve`).
- **Confusing the OpenAI sub-traces with the request traces** — Phoenix captures every OpenAI SDK call as its own trace by default — the embedding call for retrieval shows up as a separate trace ID alongside the `rag_query`-rooted ones. The rubric §7 evidence is the `rag_query`-rooted trace; the `summarize_traces` helper filters out the others by grouping on the root span name. Filter on `name == "rag_query"` when you want only the full request journey from the UI.
- **Forgetting that the OpenInference instrumentor is provider-agnostic** — the `OpenAIInstrumentor` patches the OpenAI SDK, which the starter uses regardless of whether `OPENAI_BASE_URL` points at the default OpenAI endpoint or at Vocareum's proxy. The tracer captures the same span data either way; the only difference is which host serves the actual chat completions call.

## What you have now

A traced ScikitDocs pipeline you wired by hand, plus three artifacts that make the workflow producible: a one-paragraph trace description from the UI, a `data/trace_evidence.md` file ready for the §7 rubric section, and a four-line diff that added a new attribute the downstream modules will read. That is the full operational surface of tracing for the LLM Ops course: instrument once, name your spans, read attributes by name, edit when the schema needs to grow.

A note on what the three exercises share, in case the framing was not obvious. Each ends in an artifact a teammate can reproduce. The trace ID list from exercise 1 is reproducible because the queries are written out — anyone can re-fire the same three and find equivalent traces in their own Phoenix. The evidence file from exercise 2 is regeneratable on every `make seed-traces` invocation, which means a reviewer who suspects the file is stale can re-run the script and diff the output against what was checked in. The diff snippet from exercise 3 is reviewable on its own and rerunnable inside the existing test suite. None of the three asks you to copy a screenshot blindly — every artifact is something the next person can verify by running the same code you ran, against the same starter, with the same `.env`. That reproducibility property is what separates production tracing evidence from a one-off screenshot from a debugging session.

Three forward references. The RAGAS evaluation layer (Module 11) wires evals on top of trace exports — the same `input.value` and `output.value` attributes you read in exercise 1 are exactly the fields RAGAS consumes for faithfulness and context-precision scoring. The cost-monitoring layer (Module 13) reads `llm.token_count.prompt` and `llm.token_count.completion` off the `generate` span to build per-request cost reports. The gateway (Module 18) puts `init_tracing()` in the FastAPI lifespan so `make serve` brings Phoenix up alongside `localhost:8080/query` — the same wiring you ran by hand from `python -i` happens automatically when the gateway boots. The latency layer (Module 26) starts from the latency-tail row in your `data/trace_evidence.md` file and works backward into streaming and caching. The instrumentation you wire in this module is upstream of all of them.
