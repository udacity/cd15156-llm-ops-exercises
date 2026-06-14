# Solution notes — Module 22 (Prompt A/B Testing)

The starter ships every production primitive this module needs: `src/optimization/ab.py` (three functions), both prompt variants at `prompts/docbot_system_{A,B}.j2`, the judge prompt at `prompts/judge.j2`, the analyzer at `scripts/ab_analyze.py`, the `make ab-simulate` and `make ab-analyze` Makefile targets, and the 16-test suite at `tests/test_ab.py`. The only new file this exercise authors is one script.

## Files these notes add

| File | Maps to |
|---|---|
| `scripts/ab_simulate.py` | Exercise 1 — 200-call sticky-by-user A/B harness; scores each answer with an LLM-as-judge faithfulness call; the docstring at the top doubles as Exercise 3's written decision |

Run after `make setup` + `make load-data`. The script writes to `data/ab_log.jsonl`; rerunning appends, so delete the file between fresh runs.

The success metric is an LLM-as-judge faithfulness call (`judge_supported`, rendering `prompts/judge.j2`), not the naive `any(s.doc_id in answer ...)` citation check. The citation check is degenerate on this corpus: the doc_ids are RST section anchors like `modules.svm.kernel-functions`, which the model never reproduces verbatim, so every call would score False and `scipy.stats.chi2_contingency` would raise on the all-zero column.

## Writeup-only deliverables (no code to commit)

### Exercise 1 — Harness run

Expected paste-into-writeup shape after `make ab-simulate`:

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
Unique client_ids (sticky effective N): 49

Variant A: 98/105 success (93.3%)
Variant B: 87/95 success (91.6%)
chi2 statistic:                0.041
p-value:                       0.8402
degrees of freedom:            1
significant at alpha=0.05:     False

metric                             A           B
------------------------------------------------
n                                105          95
mean_latency_ms                 4207        5755
p50_latency_ms                  3649        4733
mean_cost_usd                0.00045     0.00050
total_cost_usd               0.04682     0.04769
mean_completion_tokens           189         271
```

The honest one-paragraph interpretation: "Underpowered for sticky-by-user at ~50 unique clients — chi-squared p ~ 0.84 does not let us reject the null hypothesis of equivalence, and the sticky-effective-N (~50, not 200) makes the test even more underpowered than the raw call count suggests. We cannot conclude the variants are statistically different on the judge faithfulness label at this sample size."

### Exercise 3 — Written decision docstring

The docstring at the top of `scripts/ab_simulate.py` in these notes is the canonical example. It names the winning variant (A), the deciding metric (cost + latency at quality parity), the confidence caveat (sticky-effective-N), and the proposed next step (rerun at 500 unique clients). On a typical run variant B's "be expansive" answers carry ~40% more completion tokens, which tracks into ~35% higher mean latency and a modestly higher per-call cost at no judged-quality gain — so A wins on the secondary metrics.

## Verification

```bash
# After make setup + make load-data:
make ab-simulate                         # ~10-12 min on Vocareum (200 answer + 200 judge calls)
make ab-analyze                          # reads data/ab_log.jsonl, prints chi-squared + per-variant block
```

Both should run to completion without exceptions. The simulation budget is roughly 5-10 cents on `gpt-4o-mini` (the judge call per row roughly doubles the answer-only cost); an order of magnitude more if you accidentally point at `gpt-4o`.

## KNOWN-LIMITATIONs

The success metric is a single LLM-as-judge call per answer. A single judge call has variance, so a borderline answer can flip SUPPORTED/NOT_SUPPORTED between runs — Exercise 3's stretch averages three calls to quantify this. The judge fails open (scores SUPPORTED) on an empty answer or any API error so a transient proxy hiccup cannot depress the measured success rate; on a clean run that path is not exercised. Judge calls are deliberately *not* cost-logged, so the per-variant `cost_usd` aggregate reflects only the answer calls under comparison, not the measurement overhead. The exercise authors exactly one file (`scripts/ab_simulate.py`) whose source is the fenced code block in `exercise.md` step 2 of Exercise 1, with the docstring updated to satisfy Exercise 3's written-decision requirement. The full ten-question pool that the exercise's "add five to ten more" comment requests is included so the file runs standalone; learners can pare it down.
