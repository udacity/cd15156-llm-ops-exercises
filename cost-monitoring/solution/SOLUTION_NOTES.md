# Solution notes — Module 13 (Cost Monitoring)

The starter already ships `src/pricing.py`, `src/cost/tracker.py`, `src/cost/dashboard.py`, and `scripts/seed_cost_log.py`. The only new files this exercise authors are two scripts in `scripts/`. Everything else in this module is shell one-liners and writeup-only deliverables.

## Files these notes add

| File | Maps to |
|---|---|
| `scripts/cost_alert.py` | Exercise 3 Part A — rolling-baseline anomaly alert |
| `scripts/cost_budget_gate.py` | Exercise 3 Part B — `estimate_cost` + `gate` + reconciliation demo |

Run them after `make seed-cost-log` (Part A needs rows in `data/cost_log.jsonl`) and after `make load-data` (Part B's reconciliation block calls `run_pipeline`).

## Writeup-only deliverables (no code to commit)

### Exercise 1 — Seeded log breakdown

Expected paste-into-writeup shape after `make seed-cost-log`:

```
Total requests: 50
Total cost:     $0.0722
gpt-4o: 10 reqs, $0.0642 ($0.006422 avg)
gpt-4o-mini: 40 reqs, $0.0080 ($0.000199 avg)
```

Per-day breakdown (varies by run-time clock; the seed spreads across the past 24h):

```
2026-05-29: $0.0349  (24 requests)
2026-05-30: $0.0373  (26 requests)
```

Per-query-type breakdown (ranked by dollar total):

```
complex: $0.0642  (10 requests, avg $0.006422)
simple: $0.0061  (35 requests, avg $0.000174)
hallucination_check: $0.0019  (5 requests, avg $0.000380)
```

The one-paragraph note should name the costliest day and the costliest query type, and report the `complex` vs `simple` per-request cost ratio (typically ~25–40× on the seeded mix because both the rate ratio and the completion-length ratio compound).

### Exercise 2 — Twenty real queries

Expected writeup artifacts:

- Post-load `summarize` output showing total requests grew by 20 and total cost grew by some small fraction of a cent.
- `tail -5 data/cost_log.jsonl` excerpt showing real `timestamp`/`model`/`prompt_tokens` shape — visually identical to seeded rows.
- Two-sentence confirmation: "log grew by exactly 20 rows" (or 70 total if you appended to the seed).
- **Bonus paragraph** — back-of-envelope cache savings estimate. On a 20-query loop of 5 distinct questions × 4 repeats, 15 of the 20 calls have a cached prefix available (3 repeats per question × 5 questions). At ~1,200 stable system-prompt tokens per cached call and a 50% input-rate discount, the unrealized savings on `gpt-4o-mini` is roughly `15 × 1200 × $0.15/2 / 1_000_000 ≈ $0.00135` — sub-penny in absolute terms, ~15–30% of input cost in percentage terms. Two-step fix: (1) extend `TokenUsage` with `cached_tokens: int = 0`, (2) teach `compute_cost` to split `prompt_tokens` into `prompt_tokens - cached_tokens` at the full rate plus `cached_tokens` at the 50%-discounted rate.

### Exercise 3 — Production-tuning paragraph

Expected closing paragraph: production teams tune all three knobs but in different cadences. The per-request `limit_usd` is set once per route based on the price-tier the route exposes (a free-tier endpoint takes a tighter limit than a paid one) and re-audited quarterly. The `expected_output_tokens` heuristic gets recalibrated from the cost log — group recent rows by `query_type`, take the p95 of `completion_tokens`, plug that in. The `system_prompt_tokens` constant is verifiable today (tokenize `prompts/docbot_system.j2` once) but drifts as the template changes; pin it in the gate and update whenever the prompt is bumped.

## Verification

```bash
# After make setup + make load-data + make seed-cost-log:
uv run python scripts/cost_alert.py
uv run python scripts/cost_budget_gate.py
```

`cost_alert.py` prints today's cost, the rolling 7d avg, and either the WARNING line or nothing if today's cost is within 2× the rolling baseline.

`cost_budget_gate.py` prints the normal-question gate-passed line, the pathological-question refused line, and the reconciliation block comparing `estimate_cost` against `compute_cost` on a real `run_pipeline` call. Expect the estimate to land 10–30% above the actual cost on `gpt-4o-mini` simple questions (the conservative-error direction the gate wants).

## KNOWN-LIMITATIONs

None. The two authored scripts are direct transcriptions of the fenced code blocks in `exercise.md`; non-code deliverables are summarized as expected-output prose in this file rather than executable artifacts.
