# Wave 4 smoke RAG gate — SMOKE_REPORT

- Run at: 2026-05-18T14:19:43+00:00
- Collection: `scikit_docs`
- Embedding model: `text-embedding-3-small`
- Top-k: 5
- Recall floor: 0.70
- Seeded chunks present: 8

## Overall result

**PASS** — recall@5 = 1.00 (threshold ≥ 0.70)

## Per-query detail

### Query 1 — factual

> What is the default value of `n_estimators` in `RandomForestClassifier`?

- **Passed:** ✅ (min_hits=1)
- **Latency:** 1705.2 ms
- **Expected doc_id prefixes** (3):
  - `modules.ensemble.random-forests`
  - `modules.ensemble.parameters`
  - `modules.ensemble.forests-of-randomized-trees`
- **Retrieved top-5 doc_ids:**
  - `seeded.near_dup.random_forest_estimators_a`
  - `seeded.near_dup.random_forest_estimators_b`
  - `modules.ensemble.parameters`
  - `modules.ensemble.fitting-additional-trees`
  - `seeded.embed_confusion.random_forest_regressor`
- **Matches:**
  - modules.ensemble.parameters ⇐ modules.ensemble.parameters

### Query 2 — procedural

> How do I scale features before clustering them?

- **Passed:** ✅ (min_hits=1)
- **Latency:** 599.0 ms
- **Expected doc_id prefixes** (2):
  - `modules.preprocessing`
  - `modules.clustering`
- **Retrieved top-5 doc_ids:**
  - `modules.unsupervised_reduction.feature-agglomeration`
  - `modules.clustering.hierarchical-clustering`
  - `modules.clustering.overview-of-clustering-methods.p0`
  - `modules.clustering.adding-connectivity-constraints.p0`
  - `modules.clustering.hierarchical-clustering.p0`
- **Matches:**
  - modules.clustering.hierarchical-clustering ⇐ modules.clustering

### Query 3 — conceptual

> Why use stratified K-fold instead of plain K-fold for classification?

- **Passed:** ✅ (min_hits=1)
- **Latency:** 616.3 ms
- **Expected doc_id prefixes** (2):
  - `modules.cross_validation.stratified-k-fold`
  - `modules.cross_validation.computing-cross-validated-metrics`
- **Retrieved top-5 doc_ids:**
  - `modules.cross_validation.stratified-k-fold`
  - `modules.cross_validation.stratifiedgroupkfold.p0`
  - `modules.cross_validation.cross-validation-iterators-with-stratification-based-on-class-labels`
  - `modules.cross_validation.group-k-fold`
  - `modules.cross_validation.k-fold`
- **Matches:**
  - modules.cross_validation.stratified-k-fold ⇐ modules.cross_validation.stratified-k-fold

## Notes

- Hit-any semantics: a query passes if at least `min_hits` of its expected doc_id prefixes match retrieved doc_ids.
- Prefix matching is intentionally loose because real section anchors are slugified from RST titles and not fully predictable offline. The smoke gate validates the *shape* of retrieval, not exact anchor alignment.
- The 8 deliberately-seeded chunks (see `SEEDING_NOTES.md`) are present in the collection so the M11 top-k sweep has signal but the smoke floor is preserved.
