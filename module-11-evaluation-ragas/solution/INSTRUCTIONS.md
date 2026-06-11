# Module 11 — Build an Automated Evaluation Suite with RAGAS

## Setup (read first)

This starter is the ScikitDocs RAG app with the prompt loader, vector DB, RAG pipeline, in-process tracing, the RAGAS evaluation harness (golden set, four-metric stack, deprecated-API sub-metric, `top_k` sweep harness), cost monitoring, semantic caching, FastAPI gateway, guardrails, A/B testing, RAGOps watchers, and latency optimizations already wired. In this module you'll: (1) extend `data/golden_set.csv` with five new rows that pull difficulty up, (2) run `make eval-topk-sweep` and recommend a `top_k`, (3) write a two-question diagnostic report, and (4) add `--faithfulness-min` / `--context-recall-min` CLI flags to `scripts/run_eval.py` so the eval suite exits non-zero on regression (CI gate). Run `make setup`, then `make load-data` followed by `make seed-difficulty` to bring up the corpus, then follow the demo walkthrough and the exercise tasks below. The eval is a pure CLI workflow — `make serve` is not required for these tasks.

---

> The recorded demo walks through this codebase; the exercises below build on it.

# Module 11 — Exercise: Extend the Golden Set, Sweep `top_k`, Diagnose Two Failures, Wire a CI Gate

The recorded demo showed `make eval` running over the 30-row ScikitDocs golden set and how to read the four-metric output plus the deprecated-API sub-metric. These exercises make the workflow producible end-to-end. Four exercises, each ending in a small artifact: five golden-set rows that pull difficulty up, a `top_k` sweep table at 3-5-10, a two-question diagnostic report, and a CI threshold gate. Plan for twenty minutes, weighted toward the diagnostic exercise — that muscle memory is what production RAG eval pays for.

## Setup

Same as the demo. `make setup` passing, `make load-data` followed by `make seed-difficulty` reporting the corpus is in Chroma, and your `.env` carrying `OPENAI_API_KEY` plus `OPENAI_BASE_URL` if you are on Vocareum. You do not need `make serve` running for these exercises — RAGAS calls `run_pipeline` directly in-process, not through the HTTP gateway. The eval is a pure CLI workflow.

Quick sanity check before you start: run `uv run python scripts/run_eval.py --limit=3 --max-workers=1` to confirm the suite runs end-to-end in under two minutes against the first three rows of the golden set. If that prints five numbers and exits cleanly, your environment is good. If it returns NaN cells, your judge calls are timing out — check that `OPENAI_BASE_URL` matches your key prefix, and confirm `--max-workers=1` is the cap.

A budgeting note. One `make eval` against the 30-row set burns roughly 200-300 judge calls (four RAGAS metrics, each making one or more judge calls per row, plus the deprecated-API sub-metric runs in-process at $0 cost). At `gpt-4o-mini` pricing that is a few cents per run. `make eval-topk-sweep` triples that to about $0.08. Three full passes across the exercises adds up to under a dollar; on Vocareum the budget is the per-account cap, not direct spend. Keep your golden set small while you iterate.

One statistical note worth carrying into every exercise. At N=30, a Wilson 95% confidence interval on a recall@5 ≈ 0.8 point estimate is roughly ±0.14. Two metric values that differ by less than 0.10 are inside the noise floor; values that differ by 0.20 or more are signal. The golden set is teaching-sized, not production-sized — write your recommendations as directional rather than statistically conclusive.

## Exercise 1 — Extend the golden set with five new rows

The starter ships 30 rows at `data/golden_set.csv`. This exercise adds five rows authored against scikit-learn topics not already exercised, with a deliberate difficulty mix so the four-metric output produces variance.

### What to do

1. Open `data/golden_set.csv` and read three or four rows to internalize the schema. Six columns: `question`, `expected_doc_ids`, `min_hits`, `ground_truth_answer`, `query_type`, `version_sensitive`. The `expected_doc_ids` column is a `|`-separated list of scikit-learn doc anchors; the smoke gate at `scripts/smoke_gate.py` uses `|` as the inner separator because comma is reserved for CSV field separation. `min_hits` is the count of those anchors that must appear in the top-k retrieval for the row to count as recalled in the smoke check. `version_sensitive` is `true` for any answer that would have differed before scikit-learn 1.5. RAGAS itself does not consume `expected_doc_ids` or `min_hits` — those columns serve the smoke-gate script, not the RAGAS run.

2. Author five new questions about scikit-learn topics not already covered in the shipped 30 rows. Mix difficulty deliberately:
   - **Easy (≈2 rows)** — single-API lookups. "What does `LabelEncoder.fit_transform` return?" "What is the default value of `alpha` in `Ridge`?" These should score high on every metric on a healthy pipeline; their job is to anchor the upper end of the score distribution.
   - **Medium (≈2 rows)** — compositional or multi-step. "How do I one-hot-encode a categorical column inside a Pipeline?" "What is the difference between `partial_fit` and `fit` for incremental learning?" These exercise the retriever's ability to combine information from multiple chunks and the generator's ability to structure a procedural answer.
   - **Hard (≈1 row)** — version-sensitive or near-duplicate-prone. "What is the default solver for `LogisticRegression` and how has it changed across recent scikit-learn releases?" or "Which loss function does `HistGradientBoostingClassifier` use for binary classification?" These stress retrieval (you want both the modern chunk and any related version-conflict context to surface) and reveal whether the generator can produce a defensible answer when the retrieval is partially ambiguous. Set `version_sensitive: true` if the answer would have differed before scikit-learn 1.5.

3. Append the five rows to `data/golden_set.csv` with the same six-column shape. Mirror the CSV quoting conventions of the original — `expected_doc_ids` uses `|` as the inner separator inside a comma-separated cell. The simplest way is to copy two adjacent rows, then edit in place.

4. Run the eval against the extended set:

   ```
   PYTHONPATH=. uv run python scripts/run_eval.py \
     --max-workers=1 \
     --output=/tmp/learner-eval.json
   ```

   The `--golden` flag is the override at `scripts/run_eval.py:25-30` if you want to keep your additions in a separate file; passing the default reads the in-place extension. The output JSON carries the aggregate plus per-row scores plus the deprecated-API citations for every row.

5. Read the aggregate output and check the per-row scores in `/tmp/learner-eval.json`. Confirm two things. The easy rows score above 0.8 on faithfulness and answer relevancy; if they do not, your retrieval or generation has a baseline problem unrelated to your golden set. At least one hard row scores below 0.6 on at least one metric; if every row is above 0.8, your additions are too easy and the metric output has no signal. The variance is the value.

### Acceptance criterion

Five new rows appended to `data/golden_set.csv` with the same six-column schema. A JSON dump at `/tmp/learner-eval.json` with the aggregate metrics plus per-row scores. A one-paragraph note in your writeup naming the easy/medium/hard split, the aggregate scores, and confirming the spread (at least one row below 0.6 on at least one metric). If every score is above 0.9, rewrite at least one of your authored rows to be more ambiguous or to touch a version-sensitive topic until the spread appears.

### Hints

<details>
<summary>If `expected_doc_ids` doc anchors do not match scikit-learn's actual URL structure</summary>

The smoke gate uses prefix matching, not exact equality, against `expected_doc_ids`. An anchor that prefixes the chunk's actual `doc_id` is enough for the recall count. If a row's `min_hits` cannot be cleared with any reasonable anchor, lower `min_hits` to 1 — the goal is variance in the RAGAS metrics, not perfect smoke-gate coverage.
</details>

<details>
<summary>If the deprecated-API sub-metric drops below 1.0 on a row you authored</summary>

The generator's answer cited a symbol on the `DEPRECATED_APIS` allow-list. Open `/tmp/learner-eval.json` and read `deprecated_apis_citations` on that row — the listed symbols name the trip. Carry the finding into Exercise 3 as a deprecated-API failure case.
</details>

## Exercise 2 — Sweep `top_k` at 3, 5, and 10 and recommend a value

`top_k` is the lever that trades retrieval recall against precision. The starter exposes a sweep harness at `scripts/eval_topk_sweep.py`, wired to `make eval-topk-sweep` at `Makefile:58-59`. This exercise produces the sweep table and a one-sentence recommendation grounded in the recall-precision tradeoff.

### What to do

1. Run the sweep. The default sweeps `top_k=3,5,10`:

   ```
   make eval-topk-sweep
   ```

   Wall-clock is roughly ten to fifteen minutes against the 30-row golden set at `--max-workers=1` on Vocareum. Cost is about $0.08 — the suite fires the full four-metric stack at each of the three `top_k` values. The deprecated-API sub-metric runs in-process per row at no API cost.

   `EVAL_MAX_WORKERS` defaults to 1 for the same reason `make eval` does — parallel judge calls through Vocareum throttle. Override on uncontended endpoints: `make eval-topk-sweep EVAL_MAX_WORKERS=8`.

2. The script prints a markdown table to stdout, one row per `top_k`. Sample output shape (numbers will drift):

   ```
   | top_k | faithfulness | answer_relevancy | context_recall | context_precision |
   |-------|--------------|------------------|----------------|-------------------|
   | 3     | 0.83         | 0.92             | 0.71           | 0.78              |
   | 5     | 0.86         | 0.92             | 0.79           | 0.71              |
   | 10    | 0.87         | 0.91             | 0.86           | 0.60              |
   ```

   *Caption: `top_k=5` is the standard pick — clears the recall@5 ≥ 0.70 floor at acceptable precision, keeps the prompt short, stays within the Wilson 95% CI band of `top_k=10` on recall.*

   The visible curve is the lesson. The eight seeded difficulty chunks — three near-duplicates, three version-conflicts, two embedding-confusion pairs — guarantee the sweep produces variance. Without those eight, the unusually-clean scikit-learn docs corpus would hit ~0.95 recall at every `top_k` and the table would be flat. The seeded chunks are 0.2% of the corpus and the smoke gate's recall@5 ≥ 0.7 floor still holds; their job is to surface signal in the sweep.

   Save the table — pipe `make eval-topk-sweep > /tmp/topk-sweep.md` or copy the printed table into your writeup.

3. Read the table for the two-axis tradeoff:
   - **Context recall climbs with `top_k`** — more chunks in the bag means a higher chance the relevant one is in there. A recall-at-5 ≥ 0.7 anchor maps to `context_recall ≥ 0.7` in the RAGAS column, which is the LLM-judged analogue of binary-relevance recall-at-k. Check that your `top_k=5` row clears that floor. If it does not, the retrieval pipeline is below the rubric bar and the fix is upstream — chunking, embedding, or the similarity threshold.
   - **Context precision drops with `top_k`** — more chunks means more noise. The rank-weighted precision metric falls because irrelevant chunks dilute the top-k.
   - **Faithfulness rises slightly with `top_k`** because the generator has more material to ground claims against. But it tops out — past `top_k=10`, the prompt gets long enough that the model starts ignoring the tail.
   - **Answer relevancy is mostly flat** across `top_k` because the question itself does not change.

### Acceptance criterion

A four-row markdown table — header plus three `top_k` rows — copied into your writeup, with the caption's recommendation either confirmed by your numbers or revised in one sentence. If your recall at `top_k=5` falls below 0.7, say so explicitly and propose a fix (re-chunking, switching embedders, or lowering the retriever's similarity threshold). Acknowledge the ±0.14 confidence band when the gap between two `top_k` rows is smaller than 0.10.

### Hints

<details>
<summary>If `context_recall` is much higher than `context_precision` at every `top_k`</summary>

Expected shape for a doc corpus with structured RST sections — the retriever pulls at least one relevant chunk in but the long tail is noisy. The fix is a re-ranker (out of scope here); the writeup recommendation is "shrink `top_k` to the value that clears the recall floor at the highest precision available," which the sweep table answers directly.
</details>

<details>
<summary>If your sweep takes more than 20 minutes and you want to short-circuit it</summary>

`PYTHONPATH=. uv run python scripts/eval_topk_sweep.py --limit=10 --max-workers=1` runs only the first ten rows per `top_k`. Cuts wall clock by two-thirds at the cost of a wider Wilson CI (±0.25 at N=10). Useful for confirming the harness works before the full run.
</details>

## Exercise 3 — Diagnose two failures and recommend concrete fixes

Aggregate metrics tell you whether the system is healthy in the median. The diagnostic value of RAGAS lives in the per-row breakdown — the rows where one metric collapses while the others stay high. This exercise asks you to pick the two lowest-scoring questions from your Exercise 1 or Exercise 2 run, separate retrieval failure from generation failure for each (and check the deprecated-API sub-metric for the library-specific surface), and prescribe one concrete fix per question.

### What to do

1. Open the `/tmp/learner-eval.json` from Exercise 1 (or run the eval again with `--output` if you skipped that step). For each row, the JSON carries the question, the four RAGAS metric scores, the retrieved contexts, the generated answer, the ground truth, the `deprecated_apis_score`, and `deprecated_apis_citations`.

2. Sort the rows by the lowest single metric score (across all five surfaces, including the deprecated-API sub-metric). Pick the two rows with the lowest scores. If both lowest are faithfulness drops, that is fine — pick them; the diagnostic loop works the same. If one of your two is a `deprecated_apis_score = 0.0`, prioritize that pick — library-API hallucinations are the easiest to demonstrate and the most actionable.

3. For each picked row, apply the four-surface diagnostic from the demo:
   - **Retrieval failure.** `context_recall` low (below 0.5) and `context_precision` low. Fix lives upstream — chunking, embedding choice, similarity threshold.
   - **Generation failure.** `context_recall` and `context_precision` both high but `faithfulness` low. Fix lives in the prompt template, the model choice, or a downstream output guardrail.
   - **Routing failure.** `answer_relevancy` low while everything else is high. The pipeline answered a different question. Fix at the routing layer.
   - **Deprecated-API failure.** `deprecated_apis_score = 0.0` with `deprecated_apis_citations` listing the offending symbol. Fix is a stricter prompt naming the corpus version, or an output guardrail that scans for removed symbols. The same guardrail catches both faithfulness drops and deprecated-API drops, which is why the sub-metric sits alongside the faithfulness check rather than separately.

4. For each of the two picked rows, write two paragraphs.
   - **Paragraph 1.** State the question and the five metric scores (including `deprecated_apis_score`). Read the contexts and the answer side by side with the ground truth, and name the failure surface: retrieval, generation, routing, deprecated-API, or some combination. Acknowledge the Wilson 95% CI ±0.14 band when the metric score is close to a threshold.
   - **Paragraph 2.** Propose one concrete fix. "Use a re-ranker" is not concrete enough; "raise the retriever's similarity threshold from 0.0 to 0.3 to drop the noisy long tail, then re-run the sweep" is. "Improve the prompt" is not concrete enough; "add a sentence at the top of the system prompt instructing the model to refuse to recommend any API deprecated before scikit-learn 1.0" is.

5. Add a third paragraph naming where these fixes live downstream:
   - **Semantic caching.** A semantic cache will hide some generation failures by reusing past good answers for paraphrased questions. That is a Band-Aid, not a fix — the underlying failure still bites on the first unique phrasing — but in production it raises the effective faithfulness floor by avoiding fresh model calls on cached questions.
   - **Output guardrails.** A faithfulness-style output guardrail is the deployment-time backstop for the generation failures and deprecated-API drops you just diagnosed. It runs after the model and before the response, and it can refuse or rewrite answers that fail either check.
   - **RAGOps.** The eval suite you just ran is meant to live in CI as a regression gate and feed corpus-drift detection. Every change to the prompt, the model, or the retrieval configuration runs the five metrics and fails the build if any falls more than (say) two points below the baseline. That is what makes the eval suite operational rather than diagnostic.

### Acceptance criterion

A two-question diagnostic writeup in your project notes. For each question: the question text, the five metric scores, a one-paragraph surface-naming, and a one-paragraph concrete fix. A closing paragraph naming the downstream fixes (semantic caching, output guardrails, RAGOps regression gating and corpus-drift detection) by one sentence each. The writeup should be defensible — a teammate reading it in a code review should be able to trace every claim to a number in your `/tmp/learner-eval.json`. Acknowledge the ±0.14 confidence band somewhere in the writeup so the reader knows you treated the metrics as directional rather than statistically conclusive.

### Hints

<details>
<summary>If both of your lowest rows are version-sensitive questions</summary>

The seeded version-conflict chunks (4-6 in `data/seeded_chunks.jsonl`) confuse the retriever on questions whose answer depends on the indexed scikit-learn version. Pick one version-sensitive row and one that is not so you practice both surfaces.
</details>

<details>
<summary>If `context_recall` is high but the generator says "I don't know"</summary>

Generation failure where the model is over-cautious — right context, refused commit. Faithfulness high (refusing is faithful), answer relevancy low. Fix: lower the model's threshold for committing, or add a few-shot example of the same question answered correctly. An output guardrail is the deployment-time backstop.
</details>

## Exercise 4 — Configure a threshold gate for CI

Diagnostic value lives in per-row scores; operational value in a build-failing alarm. Production needs exit-zero-or-nonzero when a deploy regresses below an agreed floor. Artifact: a transcript with one healthy run (exit 0) and one degraded run (exit 2 + stderr naming the metric).

### What to do

1. Add `--faithfulness-min` and `--context-recall-min` flags to `scripts/run_eval.py`, defaulting to `None` so the gate only activates when set.

2. After the aggregate is computed, compare each set flag against the matching aggregate key. If any falls below its floor, print a stderr line naming the metric (actual vs floor) and `sys.exit(2)` — `2` so `make`'s internal-error `1` does not collide. Keep stdout unchanged.

3. Confirm a healthy gate exits 0:

   ```
   PYTHONPATH=. uv run python scripts/run_eval.py \
     --limit=5 --max-workers=1 \
     --faithfulness-min=0.70 --context-recall-min=0.65
   echo "exit=$?"
   ```

4. Force a failure. Simplest: pass `--faithfulness-min=1.1`. More realistic: delete a `data/docs/*.json` file a golden row depends on, `make load-data`, rerun at the step-3 floors. Confirm exit 2 + stderr naming the failed metric. Save both transcripts.

### Acceptance criterion

Two transcripts: healthy `exit=0`; degraded stderr names the failed metric, `exit=2`. `scripts/run_eval.py --help` exposes both flags. This gate is the entry point a CI workflow calls to fail a build on regression.

### Hints

<details>
<summary>If the degraded run still exits 0</summary>

Print parsed args at the top of `main()` to confirm `args.faithfulness_min` carries the float. Second-most-common cause: the aggregate dict key does not match RAGAS's emit — `print(aggregate.keys())` and confirm `faithfulness`/`context_recall` are exact-match strings.
</details>

## Common pitfalls

- **Golden set too easy.** If every row scores above 0.9, the suite is reporting "everything is fine" with no signal. Author compositional or version-sensitive rows until the distribution shows real variance.
- **LLM-judge cost runaway.** Each metric is one or more LLM calls per row. Use the smallest judge model that scores consistently (the starter pins `gpt-4o-mini`) and keep the golden set tight; running `make eval-topk-sweep` hourly in CI adds up.
- **`EVAL_MAX_WORKERS` contention.** Default 1 because Vocareum throttles parallel judges into NaN-producing TimeoutErrors. Raise on direct endpoints; leave at 1 on Vocareum — the wall-clock penalty buys correctness.
- **Ground truth phrasing affects faithfulness.** If your reference answer phrases a fact differently than the retrieved chunk (reference "100 estimators" vs chunk "n_estimators=100"), the judge sometimes splits hairs. Keep reference phrasing close to retrieved-chunk phrasing for rows you most want to score consistently.
- **Non-determinism across runs.** ±0.02 to ±0.05 drift per metric even at `temperature=0.0`. Run three times and report the mean for CI gating; one run is enough for diagnostic work.
- **Deprecated-API allow-list staleness.** `DEPRECATED_APIS` is pinned against scikit-learn 1.5. Re-derive from the release notes when the corpus is re-ingested at a newer version.
- **Confidence-band illiteracy.** Treating a 0.05 difference as significant when the Wilson 95% CI at N=30 is ±0.14. A 0.20 difference is signal; 0.05 is not.

## What you have now

A five-row golden-set extension, a `top_k` sweep table at 3-5-10 with a caption pinning the recall-precision tradeoff against the recall-at-5 ≥ 0.7 anchor, a two-question diagnostic writeup, and a CI threshold gate. Semantic caching raises the effective hit rate for repeated questions but does not fix underlying failure modes. Output guardrails are the deployment-time backstop for faithfulness drops and the deprecated-API surface. RAGOps turns the eval suite into a regression gate plus corpus-drift detection. The suite you just built is upstream of all of them.
