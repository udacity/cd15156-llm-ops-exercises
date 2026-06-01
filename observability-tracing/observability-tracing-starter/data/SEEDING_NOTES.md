# Deliberate-difficulty seeding — ScikitDocs corpus

Eight chunks in `seeded_chunks.jsonl` are upserted into the `scikit_docs`
Chroma collection alongside the real RST-derived sections. Every seeded
chunk carries `is_seeded: true` in its metadata. The seeding script
`scripts/seed_difficulty.py` is idempotent — re-running it upserts the
same eight IDs.

## Why seed at all?

Out-of-the-box, the scikit-learn doc corpus is unusually clean: section
titles map crisply to learner queries, and a `text-embedding-3-small`
retrieval with `top_k=5` will hit ~0.95 recall on a hand-authored golden
set. That ceiling makes the Module 11 RAGAS top-k sweep exercise — where
learners are supposed to *see* recall degrade from `k=10` down to `k=3`
and rationalize the operating point — pedagogically flat. Seeding 8
deliberately confusing chunks reintroduces signal without breaking the
floor: the smoke gate still passes recall@5 ≥ 0.7, but a real sweep
across k ∈ {3, 5, 10} now produces a visible curve.

## The eight seeded chunks

| # | Category | doc_id | Confusion mechanism |
|---|---|---|---|
| 1 | `near_duplicate` | `seeded.near_dup.random_forest_estimators_a` | Same RF n_estimators content as #2, different framing |
| 2 | `near_duplicate` | `seeded.near_dup.random_forest_estimators_b` | Pair-match to #1 — embeds very close |
| 3 | `near_duplicate` | `seeded.near_dup.standard_scaler_a` | StandardScaler explanation that overlaps the real preprocessing section |
| 4 | `version_conflict` | `seeded.version_conflict.knn_metric_old` | KNN metric advice tagged "deprecated-pre-0.22" |
| 5 | `version_conflict` | `seeded.version_conflict.knn_metric_new` | KNN metric advice tagged "current-1.x" — pair-match to #4 |
| 6 | `version_conflict` | `seeded.version_conflict.logreg_solver_old` | LogReg solver default ('liblinear' historical vs 'lbfgs' current) |
| 7 | `embedding_confusion` | `seeded.embed_confusion.random_forest_regressor` | RF *Regressor* docs that embed near RF *Classifier* queries |
| 8 | `embedding_confusion` | `seeded.embed_confusion.gradient_boosting_regressor` | GB *Regressor* docs that embed near GB *Classifier* queries |

`seed_pair_id` ties duplicates together for analysis (e.g. Module 24 RAGOps
exercise on detecting near-duplicate ingest drift).

## Categories — what each one teaches

### Near-duplicates (chunks 1–3)
Two embeddings cosine-close enough that top-k retrieval will sometimes
return one and sometimes the other for the same query. Module 11 (RAGAS)
sees this as `context_precision` fluctuating across runs; Module 24 (RAGOps)
sees this as a real candidate for a deduplication pre-ingest pass.

### Version conflicts (chunks 4–6)
Two chunks that answer the same question with **different correct
answers** depending on the scikit-learn version. The retriever has no
way to know which one matches the learner's intent; this is the
single biggest failure mode of "freeze the docs at one version" RAG and
exists in production library-doc Q&A systems too. Module 11 sees this as
`answer_correctness` swings; Module 24 demonstrates the fix (version-tagged
indices, blue/green corpus migration).

### Embedding confusion (chunks 7–8)
`RandomForestClassifier` and `RandomForestRegressor` are syntactic
near-twins; their docs embed close enough that a query about the
classifier sometimes pulls the regressor's section into top-k.
Pedagogically: shows that semantic similarity ≠ task relevance, and
motivates re-rankers / metadata filters.

## Distribution sanity-check vs corpus size

8 seeded chunks / ~4,000 real chunks ≈ 0.2% of the corpus. Empirical
target: every seeded chunk should appear in top-10 retrieval for at
least one golden-set query (otherwise the seeding does nothing) but no
single golden-set query should be **dominated** by seeded chunks
(otherwise we've broken the smoke gate). The smoke gate's recall@5 ≥
0.7 floor and the Module 11 sweep operating point are both validated against
these proportions.

## CSV / JSONL format notes

- `data/golden_set.csv` uses `|` (pipe) as the inner separator for the
  `expected_doc_ids` field. Comma is reserved for CSV field
  separation. Hit-any semantics: a query is "recalled" if **any** doc_id
  in its candidate set appears in the retrieval's top-k results, and
  the count of hits is at least `min_hits`.
- `data/negative_set.csv` and `data/adversarial_set.jsonl` carry no
  `expected_doc_ids` — they test guardrails / refusal behaviour, not
  retrieval.

## Statistical caveat (carry into Module 11)

N=30 yields a Wilson 95% confidence interval of roughly ±0.14 on a
recall@5 ≈ 0.8 point estimate. The golden set is **teaching-sized**,
not production-sized; Module 11's "calibrate top-k" exercise must frame the
sweep results as directional rather than statistically conclusive. The
fact that the sweep produces a *visible curve* (rather than a flat
0.95) is the point — the confidence band is the secondary lesson.

## Reproducibility

```bash
# After make load-data has populated the corpus:
uv run python scripts/seed_difficulty.py

# To verify the eight seeded chunks landed:
uv run python -c "
from src.config import settings; import chromadb
c = chromadb.PersistentClient(path=settings.chroma_path).get_collection('scikit_docs')
print('seeded count:', c.count(where={'is_seeded': True}))
"
```

## Cross-check of 5 golden rows against scikit-learn.org (the initial scaffolding acceptance)

Verified 2026-05-18 against `https://scikit-learn.org/stable/`:

| Row | Question | Verified against | Result |
|---|---|---|---|
| 1 | `n_estimators` default in `RandomForestClassifier` | `modules/generated/sklearn.ensemble.RandomForestClassifier.html` | ✅ 100; was 10 before 0.22 (`version_sensitive=true` correct) |
| 5 | `criterion` default in `DecisionTreeClassifier` | `modules/generated/sklearn.tree.DecisionTreeClassifier.html` | ✅ `"gini"` |
| 8 | `n_clusters` default in `KMeans` | `modules/generated/sklearn.cluster.KMeans.html` | ✅ 8 |
| 19 | StandardScaler `ddof=0` rationale | `modules/generated/sklearn.preprocessing.StandardScaler.html` | ✅ docs state "biased estimator … equivalent to numpy.std(x, ddof=0)" |
| 22 | L1 regularization & feature selection | `modules/linear_model.html` (Lasso section) | ✅ docs: "yields sparse models … can thus be used to perform feature selection" |

All five answers match the live documentation. No paraphrase drifts to the verbatim-doc form (RAGAS `context_recall` integrity preserved).
