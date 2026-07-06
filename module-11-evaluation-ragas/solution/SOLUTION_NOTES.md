# Module 11 Solution Notes

This reference build is the ScikitDocs starter with **Exercise 4 (CI threshold gate)** already applied to `scripts/run_eval.py` — the `--faithfulness-min` and `--context-recall-min` flags plus the `sys.exit(2)` gate body are in place. Run `uv run python scripts/run_eval.py --help` to see the new flags.

Exercises 1, 2, and 3 produce **prose/data artifacts** rather than code that lives in the repo. The expected outputs are described below; learners should produce their own and compare.

## Exercise 1 — Five new golden-set rows

The **solution** `data/golden_set.csv` ships this reference five-row extension already appended (35 rows total); the **starter** stays at the original 30 so learners author their own. The specific rows are per-learner in spirit — what matters is the schema and the difficulty spread. The reference five rows:

| difficulty | question (example) | version_sensitive |
|---|---|---|
| easy | What does `LabelEncoder.fit_transform` return? | false |
| easy | What is the default value of `alpha` in `Ridge`? | false |
| medium | How do I one-hot-encode a categorical column inside a Pipeline? | false |
| medium | What is the difference between `partial_fit` and `fit` for incremental learning? | false |
| hard | What is the default solver for `LogisticRegression` and how has it changed across recent scikit-learn releases? | true |

Append using the six-column shape, `|`-separated `expected_doc_ids`. On the shipped 35-row solution set a real run scores faithfulness ≈ 0.77, answer_relevancy ≈ 0.66, context_recall ≈ 0.72, context_precision ≈ 0.77, deprecated_apis = 1.00 — essentially the 30-row baseline (three-run mean in `data/eval_runs.md`), since five rows on thirty barely move the mean. The spread is what matters, and the extension delivers it: on that run two of the five added rows dipped below 0.6 — the `Ridge` `alpha` row **refused** (answer_relevancy 0.0, faithfulness 1.0) and the `LabelEncoder` row **missed retrieval** (context_recall 0.0) — while the version-sensitive `LogisticRegression` row tagged "hard" actually scored fine (lowest metric 0.67). Measured difficulty is not the same as assumed difficulty: author for spread, then let the run tell you which rows are genuinely hard. Read the shape, not the digits: answer_relevancy sits low because honest refusals score 0.0 as noncommittal (the refusal signature — see DEMO.md Walkthrough 2; only a minority of the zeros are retrieval misses).

If you inspect the per-row scores you will notice several **original** rows also read low. That is not caused by your five additions — RAGAS scores every row independently against its own question, retrieved context, and ground truth, so appending rows cannot change another row's score. Those originals were always low (the seeded-difficulty corpus produces ~7-9 refusals plus a batch of single-metric precision/recall dips), and any per-row difference from an earlier run is LLM-judge drift, the same ±0.14 wobble the aggregate carries. Two lessons for the price of one: per-row independence and judge drift.

A worked precision example from the extension itself: the `LabelEncoder.fit_transform` row scores context_precision ≈ 0.5. `LabelEncoder` is thinly covered in the user guide, so top-k retrieval returns one real LabelEncoder chunk plus topical siblings (`TargetEncoder`, `TransformedTargetRegressor`, `ColumnTransformer`); the siblings dilute precision to about half, and context_recall is low because the exact return-value fact is not in the retrieved chunks. That precision dip is the **stable, reproducible** signal — retrieval pulls the same chunks each run, so it is a retrieval/corpus-coverage gap, fixed upstream rather than in the prompt. Faithfulness and answer_relevancy on this row **drift run to run** (one run read 0.60 / 0.93, another 1.00 / 0.97) because the *generated* answer varies — sometimes grounded cleanly, sometimes adding a remembered detail the context does not support. A useful live contrast: the structural metric (precision, driven by retrieval) holds while the generation-driven metrics wobble.

## Exercise 2 — `top_k` sweep table

`make eval-topk-sweep` prints a markdown table to stdout. Reference numbers from one captured full-35 sweep at `--max-workers=8` (will drift across runs):

```
| top_k | faithfulness | answer_relevancy | context_recall | context_precision |
|-------|--------------|------------------|----------------|-------------------|
| 3     | 0.723        | 0.618            | 0.629          | 0.790             |
| 5     | 0.763        | 0.656            | 0.781          | 0.770             |
| 10    | 0.751        | 0.694            | 0.783          | 0.709             |
```

The trend (recall up, precision down) is clear across `top_k` 3→5→10. A judge call that gets throttled mid-row can still leave a cell reading `NaN`; re-run or drop to `EVAL_MAX_WORKERS=1` to fill it.

Recommendation: `top_k=5`, **confirmed against the captured sweep.** Context recall at `top_k=5` is 0.78, clearing the 0.70 floor, at 0.77 precision and the shortest prompt that does so; `top_k=3` misses the floor (0.63) and `top_k=10` barely improves recall (0.78) while precision falls to 0.71 and the prompt grows. Note for graders: RAGAS `context_recall` is noisy run to run and is not identical to the binary smoke-gate recall@5, which holds at 1.00; a single run can dip below the floor even when the sweep clears it. A learner who lands below 0.70 on their own run and defends `top_k=5` on the smoke gate, or picks `top_k=10`, has met the bar.

## Exercise 3 — Two-question diagnostic writeup

Per-learner — picks come from the learner's own `/tmp/learner-eval.json`. A real 30-row per-row capture is committed at `data/eval_reference.json` as the answer-key: open it to see an actual refusal (the `KMeans` `n_clusters` row scores answer_relevancy 0.0, context_recall 0.0, faithfulness 1.0 — retrieval missed, the bot refused honestly), a generation failure, and the out-of-scope negative row side by side. Expected structure: two questions × (question text, five metric scores, surface-naming paragraph, concrete-fix paragraph), then one closing paragraph naming the downstream fixes (semantic caching, output guardrails, RAGOps regression gating and corpus-drift detection). The "concrete fix" bar is "would a teammate know what file to edit and what change to make" — not "improve the prompt" but "add line X to `prompts/docbot_system.j2`."

## Exercise 4 — CI threshold gate

**Applied.** See `scripts/run_eval.py:57-76` (CLI flag declarations) and `scripts/run_eval.py:110-125` (post-aggregate gate body). Healthy invocation:

```bash
PYTHONPATH=. uv run python scripts/run_eval.py \
  --limit=5 --max-workers=4 \
  --faithfulness-min=0.70 --context-recall-min=0.65
echo "exit=$?"
```

Should print the aggregate and `exit=0`. Forcing a failure with `--faithfulness-min=1.1` should produce `FAIL: faithfulness=<measured> below floor 1.1` on stderr and `exit=2` (the measured value is whatever the run reports — roughly 0.75 on the full-30 3-run mean; a `--limit=5` subset drifts wider).
