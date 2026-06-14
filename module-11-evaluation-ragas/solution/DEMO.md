> A walkthrough of the codebase you'll work with. See INSTRUCTIONS.md for the exercise tasks.

# Demo: Run RAGAS Over ScikitDocs and Read the Metric Breakdown

The eval surface is the golden set, retrieval metrics, generation metrics, and the LLM judge sitting underneath the reference-free ones. This demo wires that surface into running code in the ScikitDocs starter. You will fire `make eval` against the 30-row golden set, read the four RAGAS scores plus the deprecated-API sub-metric, and walk through one low-scoring row to separate a retrieval failure from a generation failure.

One statistical note up front. At N=30, a Wilson 95% confidence interval on a recall@5 ≈ 0.8 point estimate is roughly ±0.14. Two metric values that differ by 0.05 are inside the noise floor; two that differ by 0.20 are signal. The eval suite is teaching-sized, not production-sized — read the numbers as directional rather than statistically conclusive.

## Setup

The demo assumes `make setup` has run, `.env` carries `OPENAI_API_KEY` (and `OPENAI_BASE_URL=https://openai.vocareum.com/v1` if you are on Vocareum), and the corpus is loaded via `make load-data` followed by `make seed-difficulty`. RAGAS reuses the same OpenAI client the application is built on — `OPENAI_BASE_URL` is read by `settings.openai_base_url` at `src/config.py:30` and threaded into both the judge LLM and the embedder at `src/evaluation/run_eval.py:116-178`. Budget for five to ten cents per `make eval` on `gpt-4o-mini`; `make eval-topk-sweep` triples that.

## Walkthrough 1 — How the eval suite is wired

`data/golden_set.csv` is the ground truth. Six columns: `question`, `expected_doc_ids`, `min_hits`, `ground_truth_answer`, `query_type`, `version_sensitive`. Thirty rows mix factual lookups (defaults, kernels, distance metrics), procedural questions, and a small number of version-sensitive rows that test whether retrieval pulls the 1.5-era chunk rather than an older one.

`src/evaluation/run_eval.py:50-55` declares the metric stack:

```python
DEFAULT_METRICS = [
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
]
```

The four-quadrant view. Faithfulness catches hallucination. Answer relevancy catches off-topic answers. Context recall catches incomplete retrieval. Context precision catches noisy retrieval. The starter pins `ragas==0.4.3` — these four are the ones that score consistently on that pin.

`_build_llm` at `src/evaluation/run_eval.py:116-153` is the RAGAS LLM wrapper. It carries two corrections to RAGAS defaults plus one anchor. `max_tokens=8192` because the default cap truncates RAGAS's statement-extraction prompts on compositional questions. `bypass_n=True` because RAGAS 0.4.x mutates a shared `n` attribute under default concurrency — 16 coroutines race for it, the loser sees `n=1`, RAGAS warns `LLM returned 1 generations instead of requested 3`, and downstream metrics go NaN. The fallback path issues sequential `n=1` calls per row, costs ~3× the API calls, but smaller per-call latency on Vocareum makes wall-clock comparable. Temperature is pinned to `constants.JUDGE_TEMPERATURE = 0.0` — the judge is a comparator, not a generator.

`build_eval_dataset` at `src/evaluation/run_eval.py:86-114` runs each golden question through `run_pipeline` and collects the four columns RAGAS needs into a `datasets.Dataset`. It rewrites the starter's `ground_truth_answer` column to the RAGAS-canonical `ground_truth` key, and forwards `top_k` so `scripts/eval_topk_sweep.py` can call this same code path at different sweep values without editing pipeline source.

The deprecated-API sub-metric lives at `src/evaluation/deprecated_apis.py`. The file holds an eight-entry `DEPRECATED_APIS` allow-list — each entry names a symbol the scikit-learn 1.5 release notes flag as removed (`sklearn.preprocessing.Imputer`, `sklearn.cross_validation`, `sklearn.grid_search`, `sklearn.externals.joblib`, the `normalize=True` argument, and a few more), with the deprecation version, replacement, and a one-line note. `score_deprecated_apis(answer)` returns 1.0 when no deprecated symbol is cited and 0.0 when at least one is; `find_deprecated_citations(answer)` returns the offending symbol names so the per-row diagnostic loop sees which API tripped the check. Binary on purpose — a documentation Q&A bot citing a removed symbol is a correctness failure regardless of how many other claims are correct, and RAGAS faithfulness can score the answer as faithful when the retrieved chunk happens to include an old example.

## Walkthrough 2 — Fire `make eval` and read the output

With the corpus loaded and the `.env` in place:

```
make eval
```

The target at `Makefile:54-55` resolves to `uv run python scripts/run_eval.py --max-workers=$(EVAL_MAX_WORKERS)`. The default `EVAL_MAX_WORKERS=8` is set at `Makefile:16`. It runs eight judge calls in parallel through the Vocareum proxy, which roughly halves the wall-clock and still scores clean. RAGAS's own default of 16 over-saturates the proxy and produces NaN cells, so don't raise it that far. If a contended endpoint returns NaN, drop to `make eval EVAL_MAX_WORKERS=1`.

Expected output (numbers drift across runs at RAGAS's internal sampling temperatures):

```
Evaluating 30 questions...
[ragas progress bars]

Aggregate metrics:
  faithfulness: 0.86
  answer_relevancy: 0.92
  context_recall: 0.79
  context_precision: 0.71
  deprecated_apis: 1.00
```

Read the four RAGAS numbers. Faithfulness 0.86 says roughly 86% of claims in the generated answers were supported by retrieved context — a drop is the hallucination alarm. Answer relevancy 0.92 says the answers are on-topic. Context recall 0.79 is the RAGAS analogue of a recall-at-5 ≥ 0.7 floor. Context precision 0.71 says relevant chunks are landing near the top with some long-tail noise.

The fifth number is scikit-learn-specific. `deprecated_apis: 1.00` says no answer cited a removed symbol. When it drops below 1.0, at least one row's answer mentioned something on the allow-list — `sklearn.cross_validation`, `sklearn.preprocessing.Imputer`, the `normalize=True` argument. RAGAS faithfulness would not catch it: if the retrieved chunk happened to include an old example, the generator's answer can be technically faithful to the chunk while still recommending a removed API. The sub-metric is the second pair of eyes. NaN in any column means a judge call failed mid-row, almost always rate-limit throttling. The default of 8 stays clear of it; drop to `EVAL_MAX_WORKERS=1` if a contended endpoint still shows NaN.

## Walkthrough 3 — Diagnose a low-scoring row

Aggregate metrics are the headline. The diagnostic value comes from drilling into the per-row table. Re-run with `--output` to save per-row scores:

```
PYTHONPATH=. uv run python scripts/run_eval.py --max-workers=1 --output /tmp/eval.json
```

`/tmp/eval.json` carries the aggregate plus a `rows` array — each entry has the question, the four RAGAS scores, the retrieved contexts, the generated answer, the ground truth, the `deprecated_apis_score`, and `deprecated_apis_citations` side by side. Sort by lowest single metric, open the row, read the fields together.

The diagnostic loop has three primary signatures plus one library-specific one. **Retrieval failure**: `context_recall` low (below 0.5), often `context_precision` low too. Fix upstream — chunking, embeddings, similarity threshold. **Generation failure**: `context_recall` high, `faithfulness` low. The right context was there and the generator ignored it or invented something. Fix the prompt, the model, or an output guardrail. **Routing failure**: `answer_relevancy` low while everything else is high. The pipeline answered a different question. Fix the classifier. **Deprecated-API failure**: `deprecated_apis_score = 0.0`. Read `deprecated_apis_citations` for the offending symbol; the generator was working from outdated training data or an old retrieved example.

A concrete example. The lowest-faithfulness row is "What is the default solver for `LogisticRegression` in scikit-learn 1.5?" Faithfulness 0.4, answer relevancy 0.85, context recall 0.65, context precision 0.55, deprecated_apis 0.0, citations `["sklearn.cross_validation"]`. The contexts include the modern LogisticRegression doc section. The answer correctly names `lbfgs` as the solver but bolts on a paragraph suggesting `sklearn.cross_validation` for hyperparameter search. Retrieval was partially right (hence recall 0.65), the generator stayed on-topic (relevancy 0.85), but it added a deprecated suggestion the retrieved context did not support — hence the faithfulness drop and the deprecated-API hit. The fix is a stricter prompt ("refuse to recommend any API deprecated before scikit-learn 1.0") or a downstream output guardrail that scans for removed symbols.

A second example. "How many components does `PCA` keep when `n_components` is not specified?" Faithfulness 0.5, answer relevancy 0.9, context recall 0.8, context precision 0.7, deprecated_apis 1.0. The retrieved contexts include both the modern PCA chunk and one of the seeded version-conflict chunks. The answer says PCA keeps "all components" — technically correct but ambiguous, missing the `min(n_samples, n_features)` that the reference answer requires. Generation-side phrasing issue, not retrieval; the right chunks were there. Fix: a few-shot example in the prompt that anchors the expected phrasing for distribution-parameter answers.

Run the suite, read the five numbers, sort by the metric that dropped most, open the lowest-scoring row, map the fix to the surface — that is the rubric-aligned diagnostic loop. The N=30 Wilson 95% confidence band means a single row's low score is suggestive rather than conclusive; if the same shape repeats across two or three rows, you have signal worth acting on. The exercises take this further: extend the golden set with five new rows that pull difficulty up, sweep `top_k` at 3, 5, and 10 with `make eval-topk-sweep`, and walk two diagnosed cases into concrete fixes.

---

