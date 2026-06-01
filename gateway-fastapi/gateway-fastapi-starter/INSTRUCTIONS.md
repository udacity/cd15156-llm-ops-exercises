# Module 18 — Build an LLM Gateway with FastAPI

## Setup

This starter is the ScikitDocs RAG app — a Q&A assistant for the scikit-learn library — with the full instrumentation stack already wired: prompt loader and Jinja templates, Chroma vector store, RAG pipeline (`run_pipeline`), Phoenix tracing, RAGAS evaluation harness, cost monitoring, semantic answer cache, FastAPI gateway, guardrails, A/B testing, RAGOps watcher, and a streaming endpoint. In this module you will read the gateway top to bottom, fire requests through it, then put muscle on three of the gateway architecture's named patterns — tier routing, retries with backoff and jitter, and multi-provider abstraction — directly into the starter.

Bring up the environment before you start:

```bash
uv sync
cp .env.example .env          # add your OPENAI_API_KEY (or Vocareum voc- key);
                              # set OPENAI_BASE_URL=https://openai.vocareum.com/v1 on Vocareum
make load-data                # ~45–60s cold, ~5s warm; ~$0.10 in embeddings
```

Smoke-check the pipeline before any gateway work:

```bash
uv run python -c "from src.pipeline import run_pipeline; print(run_pipeline('What kernel does SVC use by default?').answer[:80])"
```

If that returns a grounded answer about `rbf`, you are ready. Follow the demo walkthrough first, then work through the three exercises in order.

---

# Demo — Read the ScikitDocs Gateway, Fire a Request, Trace the Env

The gateway concept module named the architecture in four moves — secrets, observability, rate limiting, routing — and stated the bet: every cross-cutting concern that would otherwise live in N services lives in the gateway. This demo turns that picture into running code in the ScikitDocs starter. You will read the gateway top to bottom (it is small and that is the point), fire a query through it and watch the response shape match what the concept module promised, then trace the `OPENAI_BASE_URL` env var from `.env` to the OpenAI client construction sites so the Vocareum-or-direct switch is concrete. The retry and rate-limit patterns come up in the exercises; the demo's job is to make the request flow and the secret seam legible.

## Setup

With `make setup` complete and `make load-data` reporting the scikit-learn corpus is in Chroma, fire `make serve` in a separate terminal. The target resolves to `uv run uvicorn src.gateway.app:app --reload --reload-dir src --port 8080` — the server binds `localhost:8080` (`8000` is reserved for downstream services like vLLM that speak ChatCompletions). The Phoenix tracer started by the observability module boots at `localhost:6006` as a side effect of the lifespan startup. Your `.env` carries the two values the gateway cares about:

```
OPENAI_API_KEY=voc-...           # or sk-... on a direct OpenAI account
OPENAI_BASE_URL=https://openai.vocareum.com/v1   # omit on a direct OpenAI account
```

Walkthrough 3 traces these end-to-end. The short version: `OPENAI_BASE_URL` empty means "use the SDK default"; the Vocareum URL routes through the sandbox proxy that accepts `voc-` keys.

## Walkthrough 1 — Read the gateway top to bottom

Open `src/gateway/app.py`. The file is roughly 55 lines and most of it is comments. The two functions that matter are `lifespan` (lines 24–38) and `create_app` (lines 41–53). Read them in that order.

`lifespan` is the FastAPI startup-and-shutdown hook, written as an `@asynccontextmanager`. Code before `yield` runs once on startup; code after runs once on shutdown. The starter uses startup to call `init_tracing()` from the observability module — that one call boots the embedded Phoenix UI, registers the OpenTelemetry tracer provider, and auto-instruments the OpenAI SDK so every chat-completions and embeddings call emits a span. Shutdown calls `flush()` so in-flight spans drain before the process exits. The `try` / `finally` around `yield` is the pattern that makes `flush()` run even on an error-path shutdown.

`create_app()` is the application factory. Two lines do the wiring: `include_router(api_router)` mounts `src/gateway/routes.py` (`POST /query`, `GET /health`) and `include_router(cost_router)` mounts `src/cost/dashboard.py` (`GET /cost-dashboard`, the cost-monitoring module's deliverable). The FastAPI tutorial calls this pattern "bigger applications" — one app, many routers, one router per package. The app file is the wiring layer.

Open `src/gateway/routes.py` next. The `QueryRequest` model at lines 45–58 is the gateway's first defense — `question` is capped at 4,000 characters and `top_k` is bounded `[1, 20]`. Pydantic rejects oversized bodies with HTTP 422 before any embedding or LLM call. The `POST /query` handler at lines 63–128 dispatches through the guardrail stack (rate limit, prompt injection, system-prompt leak, PII redaction, then `route_query`, then hallucination check). The optional `X-Client-Id` header threads in via `Header(alias=constants.CLIENT_ID_HEADER)` and is forwarded to `route_query` as a typed kwarg. That kwarg matters: the A/B testing module will wrap `route_query` rather than the FastAPI route, and the wrapper's job becomes trivial because the route handler is doing nothing else with the value.

Open `src/gateway/router.py`. Six lines do the dispatch. `classify(question)` at line 66 returns `simple` or `complex`. `select_model(query_type)` at line 67 maps to `settings.model_simple` (gpt-4o-mini) or `settings.model_complex` (gpt-4o). `lookup(question)` at line 69 checks the semantic cache and returns immediately on a hit. On miss, `traced_pipeline(question, ...)` at line 73 runs the retrieval-and-generation through the Phoenix span context, `store(question, response)` writes the answer back to the cache, and `log_request(...)` at line 75 appends a JSONL row to `data/cost_log.jsonl`. The conditional log on miss is deliberate: cache hits did not make an LLM call, so charging them a cost row would distort the dashboard. The composition wires every previously-shipped capability into one ten-line function — that convergence is what the concept module promised the gateway would be.

Finally `src/gateway/classifier.py`. The classifier is an LLM self-classification — gpt-4o-mini reads the user query and returns `{"classification": "simple" | "complex", "reasoning": "..."}` via OpenAI's JSON mode. The fall-through at lines 71–80 — bad JSON or unexpected label falls to `complex` — is the conservative default the concept module named: when in doubt, pay a bit more for the safer route. The client construction at line 59 is the Vocareum seam Walkthrough 3 returns to.

That is the entire gateway. Five files, roughly 250 lines of code excluding comments. The concept module's architecture diagram becomes legible inline.

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

One JSONL row with `model: "gpt-4o-mini"`, the same `prompt_tokens` and `completion_tokens` from the response body, and the same `cost_usd`. The cost-monitoring dashboard reads from this file; the gateway is what writes to it. Pass the same question a second time and the response now has `"cached": true` and the cost log line count does *not* grow — the cache short-circuited the call and the conditional `log_request` saw `response.cached` was already set and skipped the write.

Fire a complex query to see the tier switch:

```
curl -s -X POST http://localhost:8080/query \
  -H 'content-type: application/json' \
  -d '{"question": "When should I prefer GradientBoostingRegressor over RandomForestRegressor, and how do their hyperparameter sensitivities differ?", "top_k": 5}' \
  | python -m json.tool | grep -E 'model|cost_usd'
```

The `model` field reads `gpt-4o` and `cost_usd` jumps roughly twentyfold because the per-token rate is ~17× the `gpt-4o-mini` rate and the completion is typically two to three times longer on a comparison prompt. That is the tiered-routing decision the cost-monitoring module framed and the gateway enforces in two lines at `router.py`. Tail the cost log again to see the new row at the higher model and cost — the dashboard is now reflecting a mixed-tier workload.

One more curl exercises the `X-Client-Id` header the starter wires for the A/B testing contract:

```
curl -s -X POST http://localhost:8080/query \
  -H 'X-Client-Id: jeff@example.com' \
  -H 'content-type: application/json' \
  -d '{"question": "Default kernel for SVC?"}' | python -m json.tool | grep -E 'model|cached'
```

The response shape is unchanged — `X-Client-Id` is metadata the gateway passes through to `route_query` without consuming. The A/B testing module reads the same header for sticky-by-user variant assignment, hashing the identifier modulo the number of variants so a given user keeps landing on the same arm. The contract test at `tests/test_smoke.py::test_x_client_id_header_passes_through_to_router` pins the plumbing — run `make test` and confirm it passes. The A/B testing module lands the consumer side; this module ships the plumbing.

## Walkthrough 3 — Trace the env var from `.env` to the OpenAI client

Open `src/config.py`. Line 18 is `load_dotenv()` — pydantic-settings populates `Settings` from `.env` but does not export back to `os.environ`, so libraries that bypass `Settings` (the RAGAS internal `OpenAI()` client in the evaluation module, for example) would otherwise miss your values. The explicit `load_dotenv()` bridges that gap.

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

# Exercises — Extend the Classifier, Wrap with Retry, Add a Second Provider

The demo wired the gateway into your hands. These exercises put muscle on three of the gateway architecture's named patterns — tier routing, retries with backoff and jitter, and multi-provider abstraction — directly into the ScikitDocs starter. Plan twenty minutes, roughly seven each on the first two exercises and six on the third. The third exercise is structurally the heaviest (a new adapter file) but the smallest in linecount because most of the work is reusing the shape the starter already enforces.

## Setup

Same as the demo. With `make verify` passing and `make load-data` reporting the corpus is in Chroma, your `.env` carrying `OPENAI_API_KEY` plus `OPENAI_BASE_URL` when you are on Vocareum. All three exercises hit the gateway via `make serve` on `localhost:8080` and need a working OpenAI key; Exercise 3 stubs the Anthropic provider so you do not need an Anthropic key for the acceptance test. Budget under a cent on Vocareum for the full set — these exercises issue maybe twenty LLM calls between them, mostly to gpt-4o-mini at sub-cent rates.

Before you start, the optional cache clear keeps Exercise 1's tier dispatch readable:

```
uv run python -c "from src.cache import clear; print('removed:', clear())"
```

A populated cache will hit the response before the classifier runs, so the `model` field in the response stays whatever was originally cached. Clearing avoids that confusion.

## Exercise 1 — Extend the classifier to a third tier

The starter's classifier returns `simple` or `complex`; the gateway routes to `gpt-4o-mini` or `gpt-4o`. A third tier — call it `premium` — gives you somewhere to dispatch long-context or high-stakes queries that benefit from a model with a larger window or stronger reasoning, while keeping the cheap default for the bulk of traffic. Adding a tier is a four-step edit: extend `Settings` with the new model name, extend the classifier's `QueryType` literal, edit the prompt template to teach the new label, and add the dispatch arm in `select_model`.

### What to do

1. Add the new model name to `Settings` at `src/config.py`. Below the existing `model_complex` and `model_simple` declarations, add:

   ```python
   model_premium: str = "gpt-4o"  # placeholder — swap to gpt-4-turbo or claude-sonnet when available
   ```

   The placeholder keeps the dispatch Vocareum-compatible (the sandbox accepts the same OpenAI models the rest of the gateway uses). When you ship outside the workspace you point this at whichever larger-context or stronger-reasoning model your account supports.

2. Extend the `QueryType` literal in `src/gateway/classifier.py`. Change the declaration from:

   ```python
   QueryType = Literal["simple", "complex"]
   _VALID_LABELS: tuple[str, ...] = ("simple", "complex")
   ```

   to:

   ```python
   QueryType = Literal["simple", "complex", "premium"]
   _VALID_LABELS: tuple[str, ...] = ("simple", "complex", "premium")
   ```

   The fallback target inside `classify` stays `complex` — when the classifier returns garbage, defaulting to the mid-tier model is the safer choice than escalating unexpected input to the priciest tier.

3. Teach the prompt template the new label. Open `prompts/classifier.j2`. The existing rubric describes `simple` (factual lookup) and `complex` (comparison or multi-step). Add a third bullet:

   ```
   - **premium**: A query with long context (multi-paragraph requirements, multiple constraints stacked), or a query where wrong answers carry high cost (deprecated-API checks, breaking changes across scikit-learn versions, parameter interactions across estimators). These benefit from a higher-capability model.
   ```

   The classifier is gpt-4o-mini self-classifying via JSON mode, so the prompt's wording does the load-bearing work. Keep the rubric language concrete — naming the *kind* of query that earns the tier beats abstract criteria like "high-stakes" alone.

4. Add the dispatch arm in `src/gateway/router.py`. Replace `select_model` with a three-branch dispatch:

   ```python
   def select_model(query_type: QueryType) -> str:
       if query_type == "premium":
           return settings.model_premium
       if query_type == "complex":
           return settings.model_complex
       return settings.model_simple
   ```

   Order matters here — `premium` checks first because it is the most specific case. The existing trace, cache, and cost-log code already takes the model name as a string, so no change is needed downstream.

5. Run the gateway with `make serve` and fire five queries through `/query` chosen to exercise the three tiers. Pick one straightforward factual lookup, two comparison queries, and two long-context or deprecated-API queries:

   ```
   queries=(
     "What is the default criterion for RandomForestRegressor?"
     "Compare GradientBoostingClassifier and RandomForestClassifier for imbalanced binary classification."
     "Walk me through choosing between l1 and l2 penalty on LogisticRegression for a sparse-feature problem."
     "What changed in StandardScaler between scikit-learn 0.24 and 1.4, and which arguments were deprecated?"
     "Explain every parameter of GridSearchCV's __init__ and how scoring interacts with refit when multiple scorers are passed."
   )
   for q in "${queries[@]}"; do
     curl -s -X POST http://localhost:8080/query \
       -H 'content-type: application/json' \
       -d "{\"question\": \"$q\", \"top_k\": 5}" \
       | python -c "import sys,json; r=json.load(sys.stdin); print(r['model'], '|', '$q'[:60])"
   done
   ```

   Expect a mix — the factual lookup goes to `gpt-4o-mini`, the comparison and recommendation queries to `gpt-4o`, and at least one of the long-context or deprecated-API queries to your `model_premium` (which is also `gpt-4o` in the placeholder, so observability comes from the cost log's `query_type` column rather than the model name in this case). The exact split depends on how the classifier reads each query — gpt-4o-mini self-classification is not deterministic across runs and a borderline query may oscillate.

### Acceptance criterion

Two artifacts. First, the diff against the four files you edited (`src/config.py`, `src/gateway/classifier.py`, `prompts/classifier.j2`, `src/gateway/router.py`). Second, the five-query run with each query's classifier-assigned tier visible — either by reading the `model` field directly when the three tiers point to distinct models, or by tailing the cost log (`tail -5 data/cost_log.jsonl`) and reading the `query_type` field. The cost log's `query_type` is set from the classifier's return value, so it reflects what the classifier decided regardless of whether two tiers happen to share a model.

A one-paragraph note explaining when you would actually pay for the premium tier in production. Long-context queries that strain the cheaper model's window are the cleanest justification; high-stakes queries where the mid-tier model's hallucination rate is unacceptable for the use case are the harder argument and the one the evaluation module's eval loop has to settle.

## Exercise 2 — Wrap the OpenAI client call with tenacity retries

The starter has no retry policy. The reasons are workload-driven — Vocareum's proxy is reliable enough on classroom traffic that adding retry is overhead, and a single-tenant dev artifact rarely sees the transient failures retries are designed for. The pattern is still load-bearing for production: any LLM client that does not retry on transient 5xx errors will surface those failures to end users one-for-one when the provider has a bad minute. tenacity is the Python standard for this (the library the concept module named explicitly) and wraps onto the OpenAI client with a decorator.

The mechanics this exercise pins are the ones the concept module's video on retries called out as load-bearing: exponential backoff so the retry storm is not immediate, jitter so the retries from many failing clients do not synchronize into a thundering herd at the recovery instant, and a tight exception filter so only transient errors retry. The AWS Builders Library article frames this as "retries are selfish" — every retry adds load to an already-struggling service, so the cost of a careless retry policy is paid by the same provider you are trying to be a good citizen to. tenacity's `wait_exponential_jitter` is exactly the right pick for the wait policy because it bakes both the exponential growth and the jitter into one combinator; you do not have to compose them manually.

### What to do

1. Add tenacity to the project. It is already a transitive dependency through several packages; the explicit add is one line in `pyproject.toml` under `[project.dependencies]`:

   ```
   "tenacity>=8.5,<10",
   ```

   followed by `uv sync` to install. Confirm with `uv run python -c "import tenacity; print(tenacity.__version__)"`.

2. Decorate the OpenAI chat-completions call site. Open `src/generator.py`. The current call at lines 72–79 is bare:

   ```python
   response = client.chat.completions.create(
       model=model,
       temperature=constants.GENERATION_TEMPERATURE,
       messages=[
           {"role": "system", "content": system_prompt},
           {"role": "user", "content": question},
       ],
   )
   ```

   Wrap it in a tenacity-decorated helper. Add to the imports at the top of the file:

   ```python
   from openai import APIConnectionError, APIStatusError
   from tenacity import (
       retry,
       retry_if_exception,
       stop_after_attempt,
       wait_exponential_jitter,
   )
   ```

   And add the helper above `generate`:

   ```python
   def _is_retryable(exc: BaseException) -> bool:
       """Retry only on transient failures — connection drops and 5xx server errors.

       Explicitly DO NOT retry on 4xx (bad request, auth, rate-limit-without-Retry-After
       semantics that tenacity can't honor cleanly). A 400 means the request was wrong;
       retrying re-sends the wrong request. A 429 with no Retry-After header is the
       provider asking for backoff — tenacity's blind exponential won't honor any header
       guidance, so leave 429 handling to a more specific code path if you add one later.
       """
       if isinstance(exc, APIConnectionError):
           return True
       if isinstance(exc, APIStatusError):
           return 500 <= exc.status_code < 600
       return False


   @retry(
       retry=retry_if_exception(_is_retryable),
       stop=stop_after_attempt(3),
       wait=wait_exponential_jitter(initial=1, max=8),
       reraise=True,
   )
   def _call_chat_completions(client, model: str, system_prompt: str, question: str):
       """Wrap OpenAI's create() with exponential-backoff retry on transient errors."""
       return client.chat.completions.create(
           model=model,
           temperature=constants.GENERATION_TEMPERATURE,
           messages=[
               {"role": "system", "content": system_prompt},
               {"role": "user", "content": question},
           ],
       )
   ```

   Then replace the bare call in `generate` with `response = _call_chat_completions(client, model, system_prompt, question)`. The `retry_if_exception(_is_retryable)` filter at decoration time delegates the per-exception decision to the helper above, so `APIConnectionError` and 5xx `APIStatusError` retry while 4xx (400 bad request, 401 auth, 429 without `Retry-After`) fail fast — the no-retry-on-4xx semantics the next step's test pins. `tenacity` also ships a broader `retry_if_exception_type((APIConnectionError, APIStatusError))` form, but that catches 4xx too and would retry pointless requests; production gateways grow toward selective filters for the same reason.

3. Test the retry behavior with a mocked flaky endpoint. Create `tests/test_retry.py`:

   ```python
   from unittest.mock import MagicMock

   import httpx
   import pytest
   from openai import APIStatusError

   from src.generator import _call_chat_completions


   def _make_5xx():
       return APIStatusError(
           "server error",
           response=httpx.Response(status_code=503, request=httpx.Request("POST", "http://test")),
           body=None,
       )


   def test_retries_succeed_after_two_5xx():
       """Two 503s then a success — tenacity should retry and the third call should return."""
       success = MagicMock()
       success.choices = [MagicMock()]
       success.choices[0].message.content = "ok"

       call_results = [_make_5xx(), _make_5xx(), success]

       def side_effect(**kwargs):
           outcome = call_results.pop(0)
           if isinstance(outcome, Exception):
               raise outcome
           return outcome

       client = MagicMock()
       client.chat.completions.create.side_effect = side_effect

       result = _call_chat_completions(client, "gpt-4o-mini", "system", "question")
       assert result.choices[0].message.content == "ok"
       assert client.chat.completions.create.call_count == 3


   def test_does_not_retry_on_400():
       """A 400 is a client error — retrying re-sends the wrong request."""
       err = APIStatusError(
           "bad request",
           response=httpx.Response(status_code=400, request=httpx.Request("POST", "http://test")),
           body=None,
       )
       client = MagicMock()
       client.chat.completions.create.side_effect = err

       with pytest.raises(APIStatusError):
           _call_chat_completions(client, "gpt-4o-mini", "system", "question")
       assert client.chat.completions.create.call_count == 1
   ```

   Run with `uv run pytest tests/test_retry.py -v`. Both tests should pass — the first shows two 5xx failures resolve into a success on the third attempt, the second shows a 4xx fails fast without retry. If the retry test takes too long to run because tenacity is delaying real-time, parametrize the decorator's `wait=` via a module-level constant the test can patch, or override with `wait=wait_exponential_jitter(initial=0.01, max=0.05)` as a test-only constructor. The production decorator's `initial=1` is the right value but slows the test suite.

### Acceptance criterion

Three artifacts. First, the diff against `src/generator.py` (the imports, the helper, the decorator, the call-site replacement) and `pyproject.toml` (the tenacity add). Second, the two passing tests from `tests/test_retry.py` showing both the retry-succeeds-on-5xx and the no-retry-on-4xx behaviors. Third, a one-paragraph note naming which production conditions actually trip the retry — provider 5xx during incidents, connection drops during deploys, brief regional network hiccups — and which conditions you deliberately do not retry — 4xx errors (the request was wrong), 429 without `Retry-After` semantics (the provider is asking for backoff and your blind exponential will not honor specific guidance). Naming the no-retry conditions is the deliverable that matters; tenacity's "retry everything" mode is what teams ship by accident, and rolling it back after a retry storm is what teaches the lesson.

## Exercise 3 — Add a thin Anthropic adapter for multi-provider routing

The concept module named multi-provider abstraction as the third payoff of running a gateway — declare model groups, route across providers, get fallback chains and cost-aware routing for free. The starter routes only OpenAI. This exercise wires the *shape* of multi-provider routing without requiring an Anthropic key on Vocareum: write a thin adapter that converts the gateway's existing input and output shape to the Anthropic Messages API and back, wire it as an alternative `provider` value on `QueryRequest`, and stub the actual Anthropic SDK call with a function that returns a canned response. The pattern is the deliverable; the credentials are not.

The two provider APIs are close enough that one adapter is small, and different enough that the conversion has named gotchas worth seeing in code. Anthropic's Messages API takes the system prompt as a top-level `system` argument; OpenAI mixes it into `messages` as the first turn with `role: "system"`. Anthropic's response carries a list of typed content blocks; OpenAI returns a single message string. Anthropic reports token counts as `input_tokens` and `output_tokens`; OpenAI uses `prompt_tokens` and `completion_tokens`. Each of those three differences shows up as a two-line conversion in the adapter below. LiteLLM and Portkey paper over these by exposing one unified surface (the OpenAI ChatCompletions shape the concept module named as the de facto standard) and converting at their internal boundary — your adapter does the same thing on a smaller scale.

### What to do

1. Add an `AnthropicProvider` adapter at `src/gateway/providers/anthropic.py`. The directory `src/gateway/providers/` is new — create the `__init__.py` alongside. The adapter has one entry point that takes the gateway's existing inputs (a model name, a system prompt, a user question) and returns the existing `(answer, TokenUsage, cost_usd)` triple:

   ```python
   """Anthropic Messages API adapter — converts gateway I/O to Anthropic's request and response shape.

   Demonstrates the multi-provider abstraction pattern the gateway concept module named.
   The starter routes only OpenAI in production; this adapter is the pattern, not the
   credentials. A live integration would import the `anthropic` SDK and replace the stub
   at `_call_anthropic` below with `anthropic.Anthropic(api_key=...).messages.create(...)`.
   """

   from src.models import TokenUsage
   from src.pricing import compute_cost


   def _call_anthropic(model: str, system_prompt: str, question: str, max_tokens: int = 1024) -> dict:
       """STUB — replace with anthropic.Anthropic(api_key=...).messages.create(...) in production.

       Returns the Anthropic Messages API response shape so the calling code can be tested
       without an Anthropic key. The real response shape from anthropic.messages.create()
       has the same `content` list of text blocks and the same `usage` dict with
       `input_tokens` and `output_tokens`.
       """
       return {
           "id": "msg_stub_0001",
           "model": model,
           "content": [{"type": "text", "text": f"[stub] answer for: {question[:40]}"}],
           "usage": {"input_tokens": 410, "output_tokens": 35},
       }


   def generate(question: str, system_prompt: str, model: str) -> tuple[str, TokenUsage, float]:
       """Provider-shaped generator with the same return contract as src.generator.generate.

       Two shape conversions happen here. On the request side, Anthropic's API takes
       `system` and `messages` separately (OpenAI mixes both into `messages`), so the
       system prompt threads in as a top-level argument. On the response side, Anthropic
       returns a list of content blocks rather than a single message string, so we
       concatenate the text-typed blocks. Token usage uses different field names
       (`input_tokens` / `output_tokens` vs OpenAI's `prompt_tokens` / `completion_tokens`)
       and we remap.
       """
       response = _call_anthropic(model, system_prompt, question)
       answer = "".join(
           block["text"] for block in response["content"] if block["type"] == "text"
       )
       usage = TokenUsage(
           prompt_tokens=response["usage"]["input_tokens"],
           completion_tokens=response["usage"]["output_tokens"],
       )
       cost = compute_cost(model, usage)
       return answer, usage, cost
   ```

   The `compute_cost(model, usage)` call assumes the model name appears in `src/pricing.py`'s rate table. For the exercise, add one row to that table for a synthetic Anthropic model — `"claude-sonnet-stub": (3.00, 15.00)` or whichever per-million-token rate Anthropic's pricing page shows on the day you read it. The number is illustrative; what matters is the shape.

2. Add a `provider` parameter to `QueryRequest` in `src/gateway/routes.py`. Below the existing `top_k` field, add:

   ```python
   provider: Literal["openai", "anthropic"] = "openai"
   ```

   and add `from typing import Literal` to the imports at the top of the file if it is not already there. The `Literal` constraint means Pydantic rejects any other value with HTTP 422 before any provider call. Defaulting to `openai` preserves the existing contract for callers who do not pass the new field.

3. Add provider dispatch in `src/gateway/router.py`. Extend `route_query` to accept a `provider` keyword and branch on it before calling the underlying pipeline:

   ```python
   def route_query(
       question: str,
       top_k: int = 5,
       *,
       model: str | None = None,
       client_id: str | None = None,
       provider: str = "openai",
   ) -> QueryResponse:
       query_type = classify(question)
       chosen_model = model or select_model(query_type)
       if provider == "anthropic":
           chosen_model = "claude-sonnet-stub"  # the synthetic model name from src/pricing.py

       hit = lookup(question)
       if hit is not None:
           return hit

       if provider == "anthropic":
           from src.gateway.providers.anthropic import generate as anthropic_generate
           from src.generator import render_system_prompt
           from src.store import query as store_query
           from src.embedder import embed_query as embed
           sources = store_query(embed(question), n_results=top_k)
           system_prompt = render_system_prompt(sources)
           answer, usage, cost = anthropic_generate(question, system_prompt, chosen_model)
           response = QueryResponse(
               answer=answer, sources=sources,
               confidence=sum(s.similarity_score for s in sources) / max(len(sources), 1),
               model=chosen_model, tokens=usage, cost_usd=cost,
           )
       else:
           response = traced_pipeline(question, top_k=top_k, model=chosen_model)

       store(question, response)
       log_request(chosen_model, response.tokens, response.cost_usd, query_type)
       return response
   ```

   The model swap when provider is `anthropic` is deliberately simple — a real implementation would route through `Settings.anthropic_model_complex` / `Settings.anthropic_model_simple` per tier, mirroring the OpenAI side. For the exercise, one synthetic name per provider keeps the diff small. The Anthropic branch reuses the starter's retrieval (embed + Chroma top-k) and the same Jinja-rendered system prompt, so the only provider-specific code is the call and the response shape conversion.

4. Update the route handler at `src/gateway/routes.py` to thread the new field:

   ```python
   return route_query(
       request.question,
       top_k=request.top_k,
       client_id=client_id,
       provider=request.provider,
   )
   ```

5. Test the round-trip:

   ```
   curl -s -X POST http://localhost:8080/query \
     -H 'content-type: application/json' \
     -d '{"question": "Default criterion for RandomForestRegressor?", "provider": "openai"}' \
     | python -m json.tool | grep -E 'model|answer'

   curl -s -X POST http://localhost:8080/query \
     -H 'content-type: application/json' \
     -d '{"question": "Default criterion for RandomForestRegressor?", "provider": "anthropic"}' \
     | python -m json.tool | grep -E 'model|answer'
   ```

   The first returns the real `gpt-4o-mini` answer with the real token counts. The second returns the stub answer with stubbed token counts. The `QueryResponse` shape is identical across both — same fields, same types, same semantics. That is the abstraction's deliverable.

### Acceptance criterion

Three artifacts. First, the new `src/gateway/providers/anthropic.py` adapter file and the directory's `__init__.py`. Second, the diffs against `src/gateway/routes.py` (the `provider` field), `src/gateway/router.py` (the dispatch), and `src/pricing.py` (the synthetic Anthropic model rate). Third, the two curl invocations side by side showing the same response shape with the model field reflecting the provider choice. A one-paragraph note discussing what changes when you swap the stub at `_call_anthropic` for the real `anthropic` SDK — the same I/O contract, the same response-shape conversion, just an actual API call with an actual key. The whole point of the adapter is that this swap is mechanical.

## Common pitfalls

A few traps that catch most learners on this module:

- **Retry semantics on non-idempotent calls.** The concept module named idempotency keys as the standard mitigation. The chat-completions calls in this exercise are read-only (the LLM does not mutate state on your end), so blind retry is safe. The moment your gateway adds tool use that mutates state — calling an external API, writing to a database, sending an email — retrying without an idempotency key double-acts. The pattern is to generate an idempotency key per request, send it as a header (the OpenAI SDK does not propagate this automatically; you would add it via `extra_headers`), and have the upstream API dedupe on it. Without that, the retry policy you just added is a foot-gun the moment your prompt becomes agentic.

- **Rate-limit middleware ordering.** When you add slowapi (not in this exercise; named in the concept module and described here for reference), the limiter middleware must register *before* the auth middleware so rate-limit-by-API-key works — the limiter needs the key in the request scope, which the auth middleware sets. Reverse the order and the limiter sees anonymous requests and rate-limits per-IP instead of per-key. The FastAPI middleware docs at `https://fastapi.tiangolo.com/tutorial/middleware/` name the stack order — middlewares run in registration order on the request, reverse order on the response. The pattern is: auth first, then rate limit, then logging.

- **Secret leakage in logs.** The concept module named "never log a key, even at debug level" as the rotation discipline. In FastAPI the trap is logging request headers at debug — `Authorization: Bearer <token>` will appear verbatim in the access log if you do not filter it. The starter does not log headers; if you add request-logging middleware as part of the gateway, filter the `Authorization`, `X-Api-Key`, and any custom auth header in the log formatter, not at the call site. The same filter applies to the cost log if you ever extend it to include user identity — hash the `X-Client-Id` value, do not log it raw.

- **Provider-specific token-count semantics.** Exercise 3's adapter remaps `input_tokens` / `output_tokens` to `prompt_tokens` / `completion_tokens`. The OpenAI and Anthropic SDKs use different field names for the same concept; a less-careful adapter would just pass `response.usage` through and the cost computation downstream would silently see zero tokens for one provider. The general rule for multi-provider abstraction: pick one canonical shape (the starter picks OpenAI's), and convert at the adapter boundary. The concept module's "one request shape regardless of backend" framing depends on this conversion happening consistently.

- **Vocareum key prefix collision.** Vocareum keys start with `voc-`; OpenAI direct keys start with `sk-`. If you accidentally ship a `voc-` key without setting `OPENAI_BASE_URL` to the Vocareum URL, the OpenAI SDK rejects the key at `https://api.openai.com/v1` with a 401 because OpenAI's auth does not recognize the `voc-` prefix. The error message says "incorrect API key", which is true but misleading — the key is right for Vocareum, wrong for OpenAI's endpoint. Always pair the key and base URL changes in `.env`.

- **The `or None` detail on `base_url`.** `OpenAI(base_url="")` is not the same as `OpenAI()`. The empty string overrides the SDK's default rather than falling back to it, and you get a connection error against the empty URL. The starter's pattern at every construction site — `base_url=settings.openai_base_url or None` — bridges that gap by treating empty-string as "use the default". When you write a new OpenAI client construction in your own code, copy the `or None` exactly; it is the smallest detail with the largest blast radius.

## What you have now

A three-tier classifier with `simple`, `complex`, and `premium` dispatch arms and a five-query run that exercises each tier. A retry-wrapped OpenAI client call with two passing tests demonstrating retry-on-5xx and no-retry-on-4xx semantics. A thin Anthropic adapter, a `provider` field on `QueryRequest`, and a side-by-side curl invocation showing the same response shape across providers. Three of the patterns the concept module named — tier routing, retry with backoff and jitter, multi-provider abstraction — wired into the ScikitDocs starter with code you wrote and tests you can rerun.

Two forward references. The next two modules are the input/output guardrails — the gateway is where they plug in because every request flows through it, and `route_query`'s existing cache-then-trace composition already names the seam each guard will land on. The A/B testing module is downstream — the `X-Client-Id` header the starter already accepts becomes the bucketing key for sticky-by-user variant assignment, and the `provider` field you added in Exercise 3 generalizes to a feature-flag-driven model choice. The chassis you wired today carries every operational concern the rest of the course bolts on. This walks ScikitDocs (a Q&A assistant for the scikit-learn library); the capstone applies the same skill to the ThirdShotHub workload.
