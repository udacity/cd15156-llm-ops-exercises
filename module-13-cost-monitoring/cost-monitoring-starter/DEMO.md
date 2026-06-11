> A walkthrough of the codebase you'll work with. See INSTRUCTIONS.md for the exercise tasks.

# Wire `compute_cost` Into the Pipeline, Log the Row, Render the Dashboard

The four-layer instrumentation stack is per-request log, rollup aggregation, alerting, and dashboard, and the two token-count sources you want are the usage object on the response and a pre-call tiktoken estimate for budget gates. This demo wires those into the ScikitDocs starter. You will read the rate table, fire a query through `run_pipeline` and watch a row land in `data/cost_log.jsonl`, then render the dashboard at `localhost:8080/cost-dashboard`. The pipeline is provided; the piece you implement on top is the pricing math, the JSONL log, and the HTML report.

## Two ways to render the dashboard

The starter's `Makefile` has a `serve` target, but it points at `src.gateway.app:app`, which isn't part of this exercise. For this demo you have two options. Either call `run_pipeline` directly in Python and call `log_request` yourself for the JSONL row, or run `make cost-dashboard` for a tiny standalone server on port 8080 that mounts just the `/cost-dashboard` route. Both produce the same dashboard. Here you do the wiring by hand so you can see each piece.

## Setup notes specific to the cost stack

The demo assumes `make setup` has run, `.env` carries `OPENAI_API_KEY` (and `OPENAI_BASE_URL=https://openai.vocareum.com/v1` if you are on Vocareum), and the corpus is loaded via `make load-data` followed by `make seed-difficulty`. The smoke check — `uv run python -c "from src.pipeline import run_pipeline; print(run_pipeline('What kernel does SVC use by default?').answer[:80])"` — is the sanity gate. If it returns a grounded answer, you are ready.

One detail to flag before you fire anything. Prompt caching changes the usage object. On OpenAI as of May 2026, `response.usage.prompt_tokens_details.cached_tokens` reports the cached prefix length when a recent identical prompt prefix exists (the auto-cache triggers at 1,024 tokens minimum per the OpenAI prompt-caching guide). On Anthropic, `usage.cache_read_input_tokens` and `usage.cache_creation_input_tokens` carry the same information for explicit `cache_control` blocks. The starter's `TokenUsage` at `src/models.py:20-28` does not yet track `cached_tokens` separately — the cost log under-resolves cache reads as plain input today, which Exercise 2's bonus addresses. Worth knowing now so the cache-aware extension in Exercise 2 is not a surprise.

## Walkthrough 1 — How the cost is computed from tokens

Open three files. They keep the cost stack composable.

`src/pricing.py` is the per-model rate table and the cost function. The whole file fits in a dozen lines:

```python
