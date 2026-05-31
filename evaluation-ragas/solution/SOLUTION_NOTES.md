# Module 11 Solution Notes

This `solution/` is the ScikitDocs starter with **Exercise 4 (CI threshold gate)** already applied to `scripts/run_eval.py` — the `--faithfulness-min` and `--context-recall-min` flags plus the `sys.exit(2)` gate body are in place. Run `uv run python scripts/run_eval.py --help` to see the new flags.

Exercises 1, 2, and 3 produce **prose/data artifacts** rather than code that lives in the repo. The expected outputs are described below; learners should produce their own and compare.

## Exercise 1 — Five new golden-set rows

The shipped `data/golden_set.csv` is unchanged in this solution because the authored rows are per-learner — what matters is the schema and the difficulty spread. A reference five-row extension:

| difficulty | question (example) | version_sensitive |
|---|---|---|
| easy | What does `LabelEncoder.fit_transform` return? | false |
| easy | What is the default value of `alpha` in `Ridge`? | false |
| medium | How do I one-hot-encode a categorical column inside a Pipeline? | false |
| medium | What is the difference between `partial_fit` and `fit` for incremental learning? | false |
| hard | What is the default solver for `LogisticRegression` and how has it changed across recent scikit-learn releases? | true |

Append using the six-column shape, `|`-separated `expected_doc_ids`. Expected post-extension aggregate: faithfulness ≈ 0.83–0.88, answer_relevancy ≈ 0.90–0.93, context_recall ≈ 0.76–0.82, context_precision ≈ 0.68–0.74, deprecated_apis = 1.00 (one of the hard rows may drop this). At least one hard row should score below 0.6 on at least one metric — that is the variance the exercise is hunting.

## Exercise 2 — `top_k` sweep table

`make eval-topk-sweep` prints a markdown table to stdout. Reference numbers (will drift across runs):

```
| top_k | faithfulness | answer_relevancy | context_recall | context_precision |
|-------|--------------|------------------|----------------|-------------------|
| 3     | 0.83         | 0.92             | 0.71           | 0.78              |
| 5     | 0.86         | 0.92             | 0.79           | 0.71              |
| 10    | 0.87         | 0.91             | 0.86           | 0.60              |
```

Recommendation: `top_k=5` — clears recall ≥ 0.70 at acceptable precision (0.71), keeps the prompt short, and lies within the Wilson 95% CI band (±0.14) of `top_k=10` on recall.

## Exercise 3 — Two-question diagnostic writeup

Per-learner — picks come from the learner's own `/tmp/learner-eval.json`. Expected structure: two questions × (question text, five metric scores, surface-naming paragraph, concrete-fix paragraph), then one closing paragraph forward-referencing M15/M20/M23/M24. The "concrete fix" bar is "would a teammate know what file to edit and what change to make" — not "improve the prompt" but "add line X to `prompts/docbot_system.j2`."

## Exercise 4 — CI threshold gate

**Applied.** See `scripts/run_eval.py:57-76` (CLI flag declarations) and `scripts/run_eval.py:110-125` (post-aggregate gate body). Healthy invocation:

```bash
PYTHONPATH=. uv run python scripts/run_eval.py \
  --limit=5 --max-workers=1 \
  --faithfulness-min=0.70 --context-recall-min=0.65
echo "exit=$?"
```

Should print the aggregate and `exit=0`. Forcing a failure with `--faithfulness-min=1.1` should produce `FAIL: faithfulness=0.86 below floor 1.1` on stderr and `exit=2`.
