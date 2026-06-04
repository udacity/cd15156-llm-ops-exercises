# Solution notes — Module 18 (Build an LLM Gateway with FastAPI)

The starter ships the gateway scaffold (lifespan + tracing init, `route_query` composition, classifier, generator, cost log, semantic cache, guardrails) from the prior module. This module exercises three extension patterns — add a tier, wrap the OpenAI call with retries, add a second provider — by editing the existing scaffold rather than authoring greenfield files.

## Files this solution edits

| File | Maps to |
|---|---|
| `src/config.py` | Exercise 1 — adds `model_premium` setting |
| `src/gateway/classifier.py` | Exercise 1 — extends `QueryType` + `_VALID_LABELS` with `premium` |
| `prompts/classifier.j2` | Exercise 1 — adds the `premium` rubric bullet + updates the output format note |
| `src/gateway/router.py` | Exercise 1 — `select_model` gains a `premium` arm; Exercise 3 — `route_query` gains a `provider` keyword + Anthropic dispatch arm |
| `pyproject.toml` | Exercise 2 — adds `tenacity>=8.5,<10` |
| `src/generator.py` | Exercise 2 — imports tenacity + openai errors, adds `_is_retryable` + `_call_chat_completions`, swaps the bare `chat.completions.create` call |
| `tests/test_retry.py` | Exercise 2 — new file; pins retry-on-5xx and no-retry-on-400 behaviors |
| `src/pricing.py` | Exercise 3 — adds `claude-sonnet-stub` rate row |
| `src/gateway/routes.py` | Exercise 3 — adds `provider` field to `QueryRequest`, threads it to `route_query` |
| `src/gateway/providers/__init__.py` | Exercise 3 — new package |
| `src/gateway/providers/anthropic.py` | Exercise 3 — new adapter file with a stubbed `_call_anthropic` |

## Note on the tenacity `retry=` filter

The decorator wires `retry=retry_if_exception(_is_retryable)` — the helper-driven filter that retries `APIConnectionError` and 5xx `APIStatusError` while letting 4xx fail fast. This matches the no-retry-on-4xx contract pinned in `tests/test_retry.py::test_does_not_retry_on_400`. The exercise prose flags the broader `retry_if_exception_type((APIConnectionError, APIStatusError))` form as a foot-gun (it would retry 400s) and prescribes the selective form as the default.

## Verification

```bash
# After uv sync (which pulls tenacity) + make load-data:
uv run pytest tests/test_retry.py -v
uv run python -c "import json; from src.gateway.classifier import QueryType, _VALID_LABELS; print(_VALID_LABELS)"
uv run python -c "from src.gateway.providers.anthropic import generate, _call_anthropic; print(_call_anthropic('claude-sonnet-stub', 'sys', 'hello'))"
```

`pytest tests/test_retry.py -v` shows both tests passing in well under a second (the production decorator's `wait=wait_exponential_jitter(initial=1, max=8)` is overridden to `initial=0.01, max=0.05` at module load time so the test suite does not block on retry backoff).

The classifier import shows `('simple', 'complex', 'premium')`; the Anthropic stub call returns the canned dict shape with `input_tokens: 410, output_tokens: 35`.

## Writeup-only deliverables (no code to commit)

### Exercise 1 — Five-query tier-dispatch run

Expected paste-into-writeup shape after the five-query loop:

```
gpt-4o-mini | What is the default criterion for RandomForestRegressor?
gpt-4o | Compare GradientBoostingClassifier and RandomForestClassifier for ...
gpt-4o | Walk me through choosing between l1 and l2 penalty on LogisticReg...
gpt-4o | What changed in StandardScaler between scikit-learn 0.24 and 1.4,...
gpt-4o | Explain every parameter of GridSearchCV's __init__ and how scorin...
```

Because the placeholder `model_premium` is also `gpt-4o`, the dispatch observability comes from the cost log's `query_type` column rather than the model name. `tail -5 data/cost_log.jsonl | python -c "import json, sys; [print(json.loads(l)['query_type']) for l in sys.stdin]"` should show a mix of `simple`, `complex`, and `premium`. The exact split is not deterministic — gpt-4o-mini self-classification oscillates on borderline queries.

Acceptance paragraph: pay for the premium tier when long-context queries strain the cheaper model's window (a multi-thousand-token requirements brief that gets truncated at the input boundary loses material the answer needs), or when the use case cannot tolerate the mid-tier model's hallucination rate on a particular query class (deprecated-API checks against scikit-learn where wrong answers ship bugs). The evaluation module's RAGAS loop is where you settle the latter argument — the dollar premium pays for itself only if the better-model hallucination rate is materially lower on the queries that actually route to the tier.

### Exercise 2 — Retry production conditions

Acceptance paragraph: retry on provider 5xx (502 / 503 / 504) during incidents, on `APIConnectionError` during deploys or brief regional network hiccups, and on `APIStatusError` with `status_code` in the 500-range. Do NOT retry on 4xx (the request was wrong; retrying re-sends the wrong request — 400, 401, 403, 404 are all immediate fail-fast), on 422 (the body failed validation; retrying with the same body fails again), or on 429 without `Retry-After` semantics (the provider is asking for backoff and tenacity's blind exponential won't honor specific header guidance — leave 429 to a more specific code path that reads `Retry-After` and sleeps accordingly). The blind "retry everything" mode is what teams ship by accident and what causes the retry storms that take down providers; the no-retry list is the deliverable that matters.

### Exercise 3 — Stub-to-real swap

Acceptance paragraph: swapping the stub at `_call_anthropic` for the real `anthropic` SDK is mechanical. Imports change from nothing to `import anthropic`. The function body changes from returning a canned dict to `return anthropic.Anthropic(api_key=settings.anthropic_api_key).messages.create(model=model, system=system_prompt, messages=[{"role": "user", "content": question}], max_tokens=max_tokens).model_dump()`. The response-shape conversion in `generate()` does not change because the real SDK returns the same `content` list of text blocks and the same `usage` dict with `input_tokens` and `output_tokens`. The downstream `compute_cost(model, usage)` call does not change because the synthetic `claude-sonnet-stub` row in `src/pricing.py` would just be replaced with the real `claude-3-5-sonnet-20241022` (or whatever model name is current) at the live published rates. The whole point of the adapter is that this swap touches one file and zero other call sites.

## KNOWN-LIMITATIONs

- **Retry test timing depends on tenacity internal monkey-patch.** `tests/test_retry.py` overrides `_call_chat_completions.retry.wait` to `wait_exponential_jitter(initial=0.01, max=0.05)` at module load time so the suite does not block on the production `initial=1` backoff. Tenacity's public API does not expose a clean re-configuration path for decorated functions, so the test reaches into the `.retry` attribute the decorator attaches. If tenacity changes that internal attribute name in a future major release the override will silently no-op and the test will become slow rather than failing — worth pinning the tenacity range tighter (`>=8.5,<9`) if the slowdown bites.

- **Anthropic adapter is wired in `route_query` via inline imports.** The local imports inside the `if provider == "anthropic":` branch keep the OpenAI-only happy path import-clean (no tenacity / providers package loaded when callers do not opt in). A production gateway would refactor this into a dispatch dict keyed on provider name so adding a third provider (Gemini, Cohere) is a single new file plus one dict entry rather than another `if` arm.

- **No live Anthropic call.** The exercise stubs the Anthropic SDK call deliberately — Vocareum does not issue Anthropic keys, and the deliverable is the adapter shape, not the credentials. The synthetic `claude-sonnet-stub` model name and the canned `{"input_tokens": 410, "output_tokens": 35}` usage mean the cost log row written on an Anthropic-routed call is illustrative rather than measured.
