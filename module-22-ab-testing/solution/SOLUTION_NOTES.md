# Solution notes — Module 22 (Prompt A/B Testing)

The starter ships every production primitive this module needs: `src/optimization/ab.py` (three functions), both prompt variants at `prompts/docbot_system_{A,B}.j2`, the analyzer at `scripts/ab_analyze.py`, the `make ab-analyze` Makefile target, and the 16-test suite at `tests/test_ab.py`. The only new file this exercise authors is one script.

## Files these notes add

| File | Maps to |
|---|---|
| `scripts/ab_simulate.py` | Exercise 1 — 200-call sticky-by-user A/B harness; the docstring at the top doubles as Exercise 3's written decision |

Run after `make setup` + `make load-data`. The script writes to `data/ab_log.jsonl`; rerunning appends, so delete the file between fresh runs.

## Writeup-only deliverables (no code to commit)

### Exercise 1 — Harness run

Expected paste-into-writeup shape after `uv run python scripts/ab_simulate.py`:

```
20/200 done
40/200 done
...
200/200 done
```

The stickiness one-liner from step 4 should print:

```
multi-variant clients: 0
```

A non-zero value means the harness is dropping `client_id` or varying the salt — re-read the loop body and confirm `pick_variant(client_id, TRAFFIC_SPLIT, salt=SALT)` is the exact call shape.

### Exercise 2 — `make ab-analyze` output

Expected output after the 200-call run (numbers will jitter, shape is fixed):

```
Loaded 200 rows from the A/B log.
Unique client_ids (sticky effective N): 50

Variant A: 87/103 success (84.5%)
Variant B: 81/97 success (83.5%)
chi2 statistic:                0.038
p-value:                       0.8451
degrees of freedom:            1
significant at alpha=0.05:     False

metric                             A           B
--------------------------------------------------
n                                103          97
mean_latency_ms                  631         984
p50_latency_ms                   604         942
mean_cost_usd               0.00012     0.00018
total_cost_usd              0.01236     0.01746
mean_completion_tokens            38          67
```

The honest one-paragraph interpretation: "Underpowered for sticky-by-user at 50 unique clients — chi-squared p ~ 0.85 does not let us reject the null hypothesis of equivalence, and the sticky-effective-N (50, not 200) makes the test even more underpowered than the raw call count suggests. We cannot conclude the variants are statistically different on success rate at this sample size."

### Exercise 3 — Written decision docstring

The docstring at the top of `scripts/ab_simulate.py` in these notes is the canonical example. It names the winning variant (A), the deciding metric (cost + latency at quality parity), the confidence caveat (sticky-effective-N), and the proposed next step (rerun at 500 unique clients).

## Verification

```bash
# After make setup + make load-data:
uv run python scripts/ab_simulate.py     # ~4-6 min on Vocareum, ~2-3 min direct OpenAI
make ab-analyze                          # reads data/ab_log.jsonl, prints chi-squared + per-variant block
```

Both should run to completion without exceptions. The simulation budget is roughly 2-5 cents on `gpt-4o-mini`; an order of magnitude more if you accidentally point at `gpt-4o`.

## KNOWN-LIMITATIONs

None. The exercise authors exactly one file (`scripts/ab_simulate.py`) whose source is the fenced code block in `exercise.md` step 2 of Exercise 1, with the docstring updated to satisfy Exercise 3's written-decision requirement. The full ten-question pool that the exercise's "add five to ten more" comment requests is included so the file runs standalone; learners can pare it down. The non-code deliverables (Exercise 2's interpretation paragraph, Exercise 3's metric-driven decision) are summarized as expected-output prose above rather than executable artifacts.
