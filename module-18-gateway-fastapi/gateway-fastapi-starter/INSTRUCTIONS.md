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

> A walkthrough of this codebase is in DEMO.md.

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
