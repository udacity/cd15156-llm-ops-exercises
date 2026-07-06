# Module 11 — captured RAGAS reference runs

Real captures from the shipped corpus state (`make load-data` + `make seed-difficulty`, 755 chunks),
`gpt-4o-mini` judge, `--max-workers=4` on Vocareum. RAGAS scores drift run to run; these are here so
you can compare your own output against real numbers, not copy them. The full per-row breakdown of
run 1 is saved alongside as `eval_reference.json` (the study artifact for Exercise 3).

## Full 30-row eval — three runs and the mean

| run | faithfulness | answer_relevancy | context_recall | context_precision | deprecated_apis |
|-----|--------------|------------------|----------------|-------------------|-----------------|
| 1   | 0.707        | 0.669            | 0.706          | 0.770             | 1.000           |
| 2   | 0.802        | 0.649            | 0.761          | 0.748             | 1.000           |
| 3   | 0.727        | 0.658            | 0.711          | 0.764             | 1.000           |
| **mean** | **0.745** | **0.659**       | **0.726**      | **0.761**         | **1.000**       |

Two things to read off this table:

1. **Answer relevancy is the lowest metric in every run** (0.65–0.67) while the other three sit in the
   0.71–0.80 band. That gap is the refusal signature: about a quarter of the golden rows are honest
   "not in the retrieved context" refusals, which RAGAS scores as noncommittal (0.0 on answer
   relevancy). It is a generation-side metric reporting a retrieval-side miss. See DEMO.md Walkthrough
   2 for the full explanation.
2. **Faithfulness swings 0.707 → 0.802 across identical runs.** That is the ±0.14 Wilson band at N=30
   in the flesh — read these numbers as directional, and prefer a mean over any single run.

## `top_k` sweep (one full-35 capture, `--max-workers=8`)

| top_k | faithfulness | answer_relevancy | context_recall | context_precision |
|-------|--------------|------------------|----------------|-------------------|
| 3     | 0.723        | 0.618            | 0.629          | 0.790             |
| 5     | 0.763        | 0.656            | 0.781          | 0.770             |
| 10    | 0.751        | 0.694            | 0.783          | 0.709             |

Context recall climbs with `top_k` (0.63 → 0.78 → 0.78); context precision falls (0.79 → 0.77 → 0.71).
`top_k=3` misses the 0.70 recall floor (0.63); `top_k=5` clears it (0.78) at 0.77 precision and the
shortest prompt that does so; `top_k=10` barely moves recall (0.78) while precision drops to 0.71.
`top_k=5` is the standard pick, confirmed by these numbers.

A judge call that gets throttled mid-row can still leave a cell reading `NaN`; re-run, or drop to
`EVAL_MAX_WORKERS=1`, to fill it.

## Independent floor

`make smoke-gate` (binary recall@5, retrieval only, separate from RAGAS `context_recall`) returns
**recall@5 = 1.00 ≥ 0.70 — PASS** on this corpus. See `data/SMOKE_REPORT.md`.
