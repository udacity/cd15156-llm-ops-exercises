# Judge calibration sweep (Exercise 2 reference solution)

Ten-row golden cohort run past the LLM-judge hallucination check, once per judge
model. Produced by `scripts/judge_calibration_sweep.py`. The knob varied here is the
**judge model** (`gpt-4o-mini` vs `gpt-4o`) on the same rubric (`prompts/judge.j2`)
at `JUDGE_TEMPERATURE = 0.0`.

> Numbers below are one real run. They drift across runs (small sample, judge
> stochasticity even at T=0). Re-run and read your own; the shape is the lesson,
> not the exact cells.

## Config A: `gpt-4o-mini` (starter default)

|                  | Judge flagged | Judge passed |
| ---------------- | ------------- | ------------ |
| Wrong-API cohort | TP = 1        | FN = 4       |
| Grounded cohort  | FP = 0        | TN = 5       |

FP rate = 0 / (0 + 5) = **0.00**  ·  FN rate = 4 / (4 + 1) = **0.80**

## Config B: `gpt-4o` (stronger judge)

|                  | Judge flagged | Judge passed |
| ---------------- | ------------- | ------------ |
| Wrong-API cohort | TP = 0        | FN = 5       |
| Grounded cohort  | FP = 1        | TN = 4       |

FP rate = 1 / (1 + 4) = **0.20**  ·  FN rate = 5 / (5 + 0) = **1.00**

## Read the per-row output before the matrices

The confusion matrices look alarming until you turn on the per-row printout, the
stdout of `scripts/judge_calibration_sweep.py`. Reproduce it yourself from
`solution/` with `PYTHONPATH=. uv run python scripts/judge_calibration_sweep.py` and
read the rows off your own screen. Every wrong-API answer is a **refusal**, not a
fabrication:

```
wrong-api 1/5  judge=FLAGGED  ans: 'Based on the documentation excerpts retrieved, there is no mention of ...'
wrong-api 2/5  judge=PASSED   ans: 'Based on the documentation excerpts retrieved, there is no information ...'
```

The questions ask about APIs that do not exist (`NormalizeAll`, `refit_index_`,
`KMeansPlus`). Retrieval found nothing to support them, so the grounded generator
declined instead of inventing. The hallucination never happened, which means the
judge had almost nothing to catch. So the "FN" cells are not escaped
hallucinations. They are refusals the judge correctly let pass.

Two more things the rows show. Three of the five **grounded** questions also came
back as refusals (`n_estimators`, the SVR kernel, the KMeans default): retrieval
missed the supporting chunk, so even the true-negative cohort is half refusals. And
the one or two `FLAGGED` verdicts that do land fall on refusal text, which is the
judge being noisy on "I don't know," not real signal.

## Interpretation

The high FN rate does not mean the judge is bad. It means the cohort never produced
the failure the judge exists to catch. A hallucination judge can only be calibrated
against actual hallucinations, and this grounded sklearn pipeline refuses rather
than confabulate. Swapping `gpt-4o-mini` for `gpt-4o` does not help, because the
problem sits upstream of the judge: with a hallucination base rate near zero on ten
rows, the model choice is measuring noise, and gpt-4o's slightly worse cells are
exactly that.

## Recommendation

Keep `gpt-4o-mini` as the judge default. The model choice is not what moves this
needle. The real result of this calibration is a finding about the **cohort**, not
the judge: to calibrate a hallucination check you have to feed it hallucinations.
Two honest ways to do that here: (1) write prompts that reliably push the model into
confabulation instead of refusal, or (2) call `check_hallucination` directly with
hand-written unsupported answers, so you test the judge in isolation from a
generator that keeps refusing. Then the FP and FN rates mean something. As shipped,
ten rows against a pipeline that refuses is not a statistically significant sample,
so it cannot support a conclusion about the judge on its own.
