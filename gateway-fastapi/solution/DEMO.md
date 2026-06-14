> A walkthrough of the codebase you'll work with. See INSTRUCTIONS.md for the exercise tasks.

# Demo — Read the ScikitDocs Gateway, Fire a Request, Trace the Env

The gateway architecture has four moves — secrets, observability, rate limiting, routing — and the bet behind them: every cross-cutting concern that would otherwise live in N services lives in the gateway. This demo turns that picture into running code in the ScikitDocs starter. You will read the gateway top to bottom (it is small and that is the point), fire a query through it and watch the response shape, then trace the `OPENAI_BASE_URL` env var from `.env` to the OpenAI client construction sites so the Vocareum-or-direct switch is concrete. The retry and rate-limit patterns come up in the exercises; the demo's job is to make the request flow and the secret seam legible.

## Setup

With `make setup` complete and `make load-data` reporting the scikit-learn corpus is in Chroma, fire `make serve` in a separate terminal. The target resolves to `uv run uvicorn src.gateway.app:app --reload --reload-dir src --port 8080` — the server binds `localhost:8080` (`8000` is reserved for downstream services like vLLM that speak ChatCompletions). The Phoenix tracer boots at `localhost:6006` as a side effect of the lifespan startup. Your `.env` carries the two values the gateway cares about:

```
OPENAI_API_KEY=voc-...           # or sk-... on a direct OpenAI account
OPENAI_BASE_URL=https://openai.vocareum.com/v1   # omit on a direct OpenAI account
```

Walkthrough 3 traces these end-to-end. The short version: `OPENAI_BASE_URL` empty means "use the SDK default"; the Vocareum URL routes through the sandbox proxy that accepts `voc-` keys.

## Walkthrough 1 — Read the gateway top to bottom

Open `src/gateway/app.py`. The file is roughly 55 lines and most of it is comments. The two functions that matter are `lifespan` (lines 24–38) and `create_app` (lines 41–53). Read them in that order.

`lifespan` is the FastAPI startup-and-shutdown hook, written as an `@asynccontextmanager`. Code before `yield` runs once on startup; code after runs once on shutdown. The starter uses startup to call `init_tracing()` — that one call boots the embedded Phoenix UI, registers the OpenTelemetry tracer provider, and auto-instruments the OpenAI SDK so every chat-completions and embeddings call emits a span. Shutdown calls `flush()` so in-flight spans drain before the process exits. The `try` / `finally` around `yield` is the pattern that makes `flush()` run even on an error-path shutdown.

`create_app()` is the application factory. Two lines do the wiring: `include_router(api_router)` mounts `src/gateway/routes.py` (`POST /query`, `GET /health`) and `include_router(cost_router)` mounts `src/cost/dashboard.py` (`GET /cost-dashboard`). The FastAPI tutorial calls this pattern "bigger applications" — one app, many routers, one router per package. The app file is the wiring layer.

Open `src/gateway/routes.py` next. The `QueryRequest` model at lines 45–58 is the gateway's first defense — `question` is capped at 4,000 characters and `top_k` is bounded `[1, 20]`. Pydantic rejects oversized bodies with HTTP 422 before any embedding or LLM call. The `POST /query` handler at lines 63–128 dispatches through the guardrail stack (rate limit, prompt injection, system-prompt leak, PII redaction, then `route_query`, then hallucination check). The optional `X-Client-Id` header threads in via `Header(alias=constants.CLIENT_ID_HEADER)` and is forwarded to `route_query` as a typed kwarg. That kwarg matters: a sticky-by-user variant assignment wraps `route_query` rather than the FastAPI route, and the wrapper's job stays trivial because the route handler is doing nothing else with the value.

Open `src/gateway/router.py`. Six lines do the dispatch. `classify(question)` at line 66 returns `simple` or `complex`. `select_model(query_type)` at line 67 maps to `settings.model_simple` (gpt-4o-mini) or `settings.model_complex` (gpt-4o). `lookup(question)` at line 69 checks the semantic cache and returns immediately on a hit. On miss, `traced_pipeline(question, ...)` at line 73 runs the retrieval-and-generation through the Phoenix span context, `store(question, response)` writes the answer back to the cache, and `log_request(...)` at line 75 appends a JSONL row to `data/cost_log.jsonl`. The conditional log on miss is deliberate: cache hits did not make an LLM call, so charging them a cost row would distort the dashboard. The composition wires every capability the starter provides into one ten-line function — that convergence is what the gateway is.

Finally `src/gateway/classifier.py`. The classifier is an LLM self-classification — gpt-4o-mini reads the user query and returns `{"classification": "simple" | "complex", "reasoning": "..."}` via OpenAI's JSON mode. The fall-through at lines 71–80 — bad JSON or unexpected label falls to `complex` — is the conservative default: when in doubt, pay a bit more for the safer route. The client construction at line 59 is the Vocareum seam Walkthrough 3 returns to.

That is the entire gateway. Five files, roughly 250 lines of code excluding comments. The architecture becomes legible inline.

## Walkthrough 2 — Fire a request, trace the response

With `make serve` up on `localhost:8080`, fire one query through `/query`:

```
curl -s -X POST http://localhost:8080/query \
  -H 'content-type: application/json' \
  -d '{"question": "What is the default criterion for RandomForestRegressor?", "top_k": 5}' \
  | python -m json.tool
```

The response is a `QueryResponse` (defined at `src/models.py:31-43`):

```
{
  "answer": "The default criterion for RandomForestRegressor is squared_error...",
  "sources": [{"doc_id": "...", "chunk_text": "...", "similarity_score": 0.87}, ...],
  "confidence": 0.84,
  "model": "gpt-4o-mini",
  "tokens": {"prompt_tokens": 1184, "completion_tokens": 42},
  "cost_usd": 0.000203,
  "cached": false,
  "trace_id": "..."
}
```

Map each field to where it came from. `model` is set by `select_model` at `src/gateway/router.py` — gpt-4o-mini because the classifier returned `simple`. `tokens` comes from `response.usage` on the OpenAI SDK return at `src/generator.py:80-84`. `cost_usd` is computed by `compute_cost(model, usage)` at `src/generator.py:85`, driven by the per-model rates in `src/pricing.py`. `cached` is `false` because this was the first time we asked. `trace_id` is the Phoenix span ID — open `localhost:6006` in a browser and search for it to see the full retrieval-and-generation trace.

Confirm the cost log row landed:

```
tail -1 data/cost_log.jsonl
```

One JSONL row with `model: "gpt-4o-mini"`, the same `prompt_tokens` and `completion_tokens` from the response body, and the same `cost_usd`. The cost dashboard reads from this file; the gateway is what writes to it. Pass the same question a second time and the response now has `"cached": true` and the cost log line count does *not* grow — the cache short-circuited the call and the conditional `log_request` saw `response.cached` was already set and skipped the write.

Fire a complex query to see the tier switch:

```
curl -s -X POST http://localhost:8080/query \
  -H 'content-type: application/json' \
  -d '{"question": "When should I prefer GradientBoostingRegressor over RandomForestRegressor, and how do their hyperparameter sensitivities differ?", "top_k": 5}' \
  | python -m json.tool | grep -E 'model|cost_usd'
```

The `model` field reads `gpt-4o` and `cost_usd` jumps roughly twentyfold because the per-token rate is ~17× the `gpt-4o-mini` rate and the completion is typically two to three times longer on a comparison prompt. That is the tiered-routing decision the gateway enforces in two lines at `router.py`. Tail the cost log again to see the new row at the higher model and cost — the dashboard is now reflecting a mixed-tier workload.

One more curl exercises the `X-Client-Id` header the starter wires for sticky-by-user routing:

```
curl -s -X POST http://localhost:8080/query \
  -H 'X-Client-Id: jeff@example.com' \
  -H 'content-type: application/json' \
  -d '{"question": "Default kernel for SVC?"}' | python -m json.tool | grep -E 'model|cached'
```

The response shape is unchanged — `X-Client-Id` is metadata the gateway passes through to `route_query` without consuming. A sticky-by-user variant assignment reads the same header, hashing the identifier modulo the number of variants so a given user keeps landing on the same arm. The contract test at `tests/test_smoke.py::test_x_client_id_header_passes_through_to_router` pins the plumbing — run `make test` and confirm it passes. The gateway ships the plumbing; the consumer side is a later concern.

## Walkthrough 3 — Trace the env var from `.env` to the OpenAI client

Open `src/config.py`. Line 18 is `load_dotenv()` — pydantic-settings populates `Settings` from `.env` but does not export back to `os.environ`, so libraries that bypass `Settings` (the RAGAS-internal `OpenAI()` client, for example) would otherwise miss your values. The explicit `load_dotenv()` bridges that gap.

Lines 27–30 declare the two fields. `openai_api_key: str = ""` is the key. `openai_base_url: str = ""` is the routing target — empty string means "use the SDK default at `https://api.openai.com/v1`", non-empty routes elsewhere. The Vocareum value is `https://openai.vocareum.com/v1`; the sandbox checks the `voc-` key prefix and meters against your course account.

The pattern at every OpenAI construction site is identical:

```python
client = OpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url or None,
)
```

`or None` is the load-bearing detail — passing an empty string to `OpenAI(base_url=...)` overrides the SDK's default rather than falling back to it. The `or None` makes empty-string equivalent to omitted. Same construction lives at `src/generator.py:70` (chat completions), `src/gateway/classifier.py:59` (the classifier), and inside `src/embedder.py` for both ingest-time and query-time embeddings. Three call sites, one secret seam.

To swap from Vocareum to a direct OpenAI account: edit `.env`, remove the `OPENAI_BASE_URL` line, restart `make serve`. To swap to a self-hosted vLLM endpoint that speaks OpenAI ChatCompletions: point `OPENAI_BASE_URL` at your vLLM URL (e.g. `http://vllm-host:8000/v1`), restart. The OpenAI SDK's `base_url` parameter is documented as the abstraction surface, and the gateway leans on it.

Retries and rate limiting are not yet wired in. Vocareum's proxy is reliable enough on classroom traffic that exponential-backoff retry is overhead rather than insurance, and a single-tenant dev artifact has no per-user rate limiting to enforce. Exercises 2 and 3 wire tenacity for retry and an Anthropic adapter for multi-provider routing so you have hands on both. The slowapi rate-limit pattern is described in Common Pitfalls without being installed, because the starter has no traffic profile that demonstrates it cleanly.

That is the demo loop. The exercises take it further: extend the classifier to a third tier so dispatch is observable in the response, wrap the OpenAI client with tenacity's exponential-backoff retry against a mock-flaky endpoint, and write a thin Anthropic adapter so the same `/query` shape works across providers with a single `provider` param flip.

---

