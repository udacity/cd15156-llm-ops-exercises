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
# USD per 1M tokens (OpenAI public pricing as of 2026-04).
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}

def compute_cost(model: str, usage: TokenUsage) -> float:
    """Return the USD cost for a completion given the model and token usage."""
    input_price, output_price = MODEL_PRICING[model]
    return (
        usage.prompt_tokens * input_price + usage.completion_tokens * output_price
    ) / 1_000_000
```

Three things to notice. The rates are dated in the docstring — `as of 2026-04` — so a future maintainer reading the file knows when to re-verify against the live OpenAI pricing page. Pricing pages move. The table keys off the same `MODEL_COMPLEX` / `MODEL_SIMPLE` invariants in `src/constants.py:25-26`; production teams version-control this dict the same way they version-control any other config that drifts. Second, an unknown model raises `KeyError` rather than silently logging $0 — that is deliberate so a typo in `.env` fails loudly. Third, the function lives outside both `src.generator` (which captures the usage) and `src.cost.tracker` (which logs it) so neither has to import the other. That separation is what makes the cost stack composable across whichever path layers on top — the synchronous `/query` route, the streaming variant, the eval loop.

The asymmetry of token pricing shows up immediately. Output tokens cost four times what input tokens cost on `gpt-4o` (ten dollars versus two-fifty per million) and four times what input costs on `gpt-4o-mini` (sixty cents versus fifteen). The dial you watch for cost regressions is output length. A verbose system prompt change adds linearly; a verbose response template adds four-times-linearly. Read the cost log with that ratio in mind.

`src/generator.py` is where the token counts come off the response and become a cost. The piece that connects pricing in is the import at the top — `from src.pricing import compute_cost` — and the final line of `generate()`:

```python
usage = TokenUsage(
    prompt_tokens=response.usage.prompt_tokens,
    completion_tokens=response.usage.completion_tokens,
)
cost_usd = compute_cost(model, usage)
return answer, usage, cost_usd
```

Before the pricing piece was connected, that line returned `cost_usd = 0.0`. Now it returns a real number. The returned tuple `(answer, usage, cost)` flows up into `run_pipeline` at `src/pipeline.py:51`, which threads `cost` into the `QueryResponse.cost_usd` field. Every caller of `run_pipeline` — the eval loop, a gateway, a streaming endpoint — now reads a real dollar number instead of a placeholder zero.

`src/cost/tracker.py` is `log_request`. It appends one JSONL row per completed call with six fields: ISO timestamp, model name, `prompt_tokens`, `completion_tokens`, `cost_usd`, and `query_type` (`simple`, `complex`, or `hallucination_check`). The function opens the file in append mode and writes one line — line-level append is the standard pattern for concurrent-write safety on POSIX, and it sidesteps the file-locking question. The schema is intentionally flat. No nesting, no optional sub-objects, one row per call. Flat rows are what makes the downstream rollup queries trivial — a one-liner over the file gives you per-model, per-day, or per-endpoint slices.

## Walkthrough 2 — Fire a query and watch the log row land

A gateway can mount the router so `log_request` is called automatically on every `/query`. Doing it by hand, you call both functions yourself:

```bash
uv run python -c "
from src.pipeline import run_pipeline
from src.cost.tracker import log_request
r = run_pipeline('What is the default value of \`n_clusters\` in \`KMeans\`?')
log_request(r.model, r.tokens, r.cost_usd, query_type='simple')
print(f'ANSWER:    {r.answer[:80]}')
print(f'MODEL:     {r.model}')
print(f'TOKENS:    prompt={r.tokens.prompt_tokens} completion={r.tokens.completion_tokens}')
print(f'COST_USD:  \${r.cost_usd:.6f}')
"
```

Representative output:

```
ANSWER:    KMeans defaults to n_clusters=8. This is documented in the
           sklearn.cluster.KMeans...
MODEL:     gpt-4o
TOKENS:    prompt=1438 completion=42
COST_USD:  $0.004015
```

Three things landed in that print. The answer cites the correct default (`8`) and names the qualified API path — that is the RAG pipeline doing its job. The dollar cost is the new piece: 1,438 input tokens on `gpt-4o` at $2.50/MTok plus 42 output tokens at $10/MTok works out to $0.00402, matching the `cost_usd` field on the response within rounding. And the `log_request` call appended a row to `data/cost_log.jsonl`. Confirm it:

```bash
tail -1 data/cost_log.jsonl
```

You should see a single JSON object with the six-field shape:

```json
{"timestamp": "2026-05-18T...", "model": "gpt-4o", "prompt_tokens": 1438, "completion_tokens": 42, "cost_usd": 0.004015, "query_type": "simple"}
```

That row is the foundation. The dashboard reads it through `load_log`, `summarize` rolls it up by model, the cost-vs-baseline math reads the same JSONL, and every alert in Exercise 3 traces back to this append. A gateway `/query` route can plumb `log_request` into the request pipeline so this happens automatically on every call; here you wired one row by hand to see the contract.

## Walkthrough 3 — Render the dashboard

Two paths to the dashboard. Pick whichever matches your environment.

**Option A — Standalone server (any environment):** in a fresh terminal, run `make cost-dashboard`. The target runs `uv run python -m src.cost.dashboard`, which spins a tiny standalone FastAPI app on `localhost:8080` with just the `/cost-dashboard` route mounted. Then from a second terminal:

```bash
curl -s http://localhost:8080/cost-dashboard | head -25
```

You should see an HTML page starting `<!doctype html>` with a totals table and a per-model breakdown. Open `http://localhost:8080/cost-dashboard` in a browser for the rendered view. Hit `Ctrl-C` on the server when you are done.

**Option B — No server, just the HTML (good for headless environments):** call the renderer directly:

```bash
uv run python -c "
from src.cost.tracker import load_log, summarize
from src.cost.dashboard import render_html
print(render_html(summarize(load_log()))[:500])
"
```

Both options produce the same HTML — the standalone server is `render_html(summarize(load_log()))` wrapped in a FastAPI route. The `APIRouter` instance at `src/cost/dashboard.py:30` is the unit a full gateway can mount. When it is mounted, the `localhost:8080/cost-dashboard` URL keeps working; you just stop needing `make cost-dashboard` because the gateway's `serve` target serves both `/query` and `/cost-dashboard`.

One more detail worth surfacing. `summarize()` at `src/cost/tracker.py:84-100` is what the dashboard renders, and it is the same function Exercise 1 calls when it walks per-model totals against a seeded log. The dict it returns has three top-level keys — `total_requests`, `total_cost_usd`, `by_model` — and the `by_model` sub-dict carries per-model `requests`, `cost_usd`, and `avg_cost_usd`. That shape is the contract every downstream visualization keys off. A production team would add per-day, per-endpoint, and per-user rollups on top by extending `summarize` with more groupers; the renderer in `dashboard.py` becomes a small set of additional tables. The starter ships the per-model slice because that is the diagnostic that matters most — which tier costs you the most — and the per-day and per-query-type slices fall out of one-liner aggregations against the same JSONL log, which Exercise 1 walks through.

That is the demo loop. The exercises take it further. Exercise 1 seeds fifty synthetic rows so you can see per-day and per-query-type rollups against a populated log without paying for fifty real queries. Exercise 2 fires twenty real queries through `run_pipeline` and watches the log grow. Exercise 3 writes a rolling-baseline alert and a pre-call tiktoken budget gate — the operational surface that turns instrumentation into protection.
