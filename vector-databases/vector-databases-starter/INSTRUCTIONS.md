# Module 05 — Configure a Vector Database with Chroma and sentence-transformers

## Setup

This starter is the ScikitDocs RAG app with the prompt loader, vector DB layer (chunker, embedder, store), RAG pipeline, tracing, evaluation harness, cost monitoring, semantic caching, gateway, guardrails, A/B testing, RAGOps, and latency optimization already wired. In this module you'll walk the three vector-DB files end-to-end (`src/chunker.py`, `src/embedder.py`, `src/store.py`), then in the exercises you'll bring up the Chroma collection from scratch, measure recall@5 against a twelve-question subset of the golden set, and swap the embedder from OpenAI's `text-embedding-3-small` to local `sentence-transformers/all-MiniLM-L6-v2` to feel the cost-vs-quality trade. Run `make setup && make load-data` from this starter's root to bring up the corpus, then follow the demo walkthrough below and the exercise tasks after the `---` separator.

---

> A walkthrough of this codebase is in DEMO.md.

# Module 05 — Exercises: Operate the ScikitDocs Vector Store

Three exercises. The first builds the collection from scratch and verifies its contents — the kind of one-time bring-up you do when joining an existing RAG project. The second runs a real recall@5 measurement against the golden set so you have a number to argue from when someone asks "is retrieval working?" The third swaps the embedder for a local sentence-transformers model and shows what the cost-and-quality trade actually looks like at this corpus size. Run them in order from the starter root.

## Exercise 1 — Build and verify the collection

Goal: starting from an empty `data/chroma/` directory, run the full ingestion pipeline, confirm Chroma holds the expected number of chunks, and inspect what one chunk looks like end-to-end.

### Steps

1. Clear the existing Chroma collection so you watch the build from zero. From the starter root:

   ```
   rm -rf data/chroma
   ```

   You are not deleting the embedding cache (`data/embedding_cache.jsonl`) — that file is keyed by SHA-256 of chunk text, so a rebuild reuses the cached vectors and avoids re-billing OpenAI. The cold-build time you observe on this run measures parse + index + upsert; embedding itself is essentially free against the cache.

2. Run the load:

   ```
   make load-data
   ```

   Read the printed timings. You should see four lines: a clone-or-checkout step (~1 second once cached locally), a parse step (~10 seconds across the RST tree), an embed step (~3–5 seconds with cache hits, ~30 seconds cold), and an upsert step (under 1 second for ~750 chunks). The final summary line ends with `done — N chunks upserted, total Ms`. Note the chunk count — you will check it against the collection in step 3.

3. Verify the collection size matches:

   ```
   uv run python -c "from src import store; print(store.get_collection().count())"
   ```

   The number should match the `chunks upserted` line from `make load-data` exactly. If it is smaller, a previous `make load-data` ran against a different `chroma_path` and the count you are reading is from a stale collection; check `settings.chroma_path` against your `.env`. If it is larger, you also ran `make seed-difficulty` previously, which adds eight deliberately-confusing chunks documented for retrieval-quality work in Module 11 — that is a feature, not a bug, and we will see those chunks in Exercise 2.

4. Inspect one chunk in detail. Chroma's `peek` returns a small sample with embeddings, documents, and metadatas attached:

   ```
   uv run python -c "
   from src import store
   sample = store.get_collection().peek(limit=1)
   print('id:', sample['ids'][0])
   print('text (first 200 chars):', sample['documents'][0][:200])
   print('metadata keys:', list(sample['metadatas'][0].keys()))
   print('embedding dim:', len(sample['embeddings'][0]))
   "
   ```

   The embedding dim must be `1536` — that is the dimensionality of `text-embedding-3-small`, locked at `constants.EMBEDDING_DIM`. The metadata keys should include `source_path`, `section_title`, `section_path`, `url`, `has_code`, `code_languages`, `xrefs`, and `scikit_learn_sha`; the list-typed fields (`code_languages`, `xrefs`) are JSON-serialised because Chroma metadata columns are scalar-only.

### What the build actually did

Read the printed timings as a five-stage pipeline, because each stage has a different failure profile and that is what makes operating a vector store sometimes surprising.

The clone-or-checkout step is the only one that talks to the public internet on a warm run, and it talks to GitHub, not to OpenAI — so it can succeed even when your `OPENAI_API_KEY` is wrong. A common bring-up failure is "clone succeeded, parse succeeded, embed failed at request one" because the key was never set; the embed stage is the first one that hits the OpenAI API. If the embed stage fails immediately with a 401, fix `.env` and re-run; the clone is cached and parse is fast, so the second run is essentially the embed-and-upsert path alone.

The parse step is the longest stage on a warm cache. RST parsing through docutils is single-threaded Python doing a lot of small allocations, and ten seconds across the full scikit-learn docs tree is the floor on this Workspace shape. The parse step also drops sections under 50 tokens — if your final chunk count is noticeably lower than the documented ~750, your token threshold or the doc-subdir list (`DOC_SUBDIRS` in `src/corpus.py`) changed.

The embed step is the one with the bill attached. The output line "embedded N new + reused M cached" tells you exactly how many API calls fired and how many were cache hits. On a first build, N is the chunk count and M is zero; on every subsequent build against the same corpus, N is zero and M is the chunk count, so the API bill goes to zero. The cache file is plaintext JSONL and small — about 50 MB for the full 750-chunk corpus — and it lives in the same directory as Chroma's storage so a single `rm -rf data/` cleans everything.

The upsert step is fast because Chroma's HNSW index appends incrementally; the only time you would see this stage slow down is when the corpus crosses ~100,000 chunks, where index rebuild costs start to matter. We are three orders of magnitude below that for ScikitDocs, so it does not surface here.

The write-version step touches `data/CORPUS_VERSION` — a four-line text file that records the scikit-learn tag, the resolved git SHA, the ingest timestamp, and the chunk count. Module 24 (RAGOps) reads this file to detect when the corpus has changed and a blue/green index swap is warranted.

### Where Chroma stores what

`data/chroma/` is a tree of small SQLite databases plus a binary HNSW index file. `peek(limit=1)` is hitting both: SQLite holds the document text and metadata, the binary file holds the vectors. The two are coupled by an internal sequence id, which is why deleting `data/chroma/` cleanly resets the collection (no orphan-file state to recover from). For exploration in a notebook, `collection.get()` returns the full document and metadata for a given id without doing a similarity search — useful for inspecting a single chunk by its `doc_id` without burning an embedding call.

### Acceptance criterion

`store.get_collection().count()` returns the same number `make load-data` printed (~747 if you skipped seed-difficulty, ~755 if you ran it), and a `peek(limit=1)` returns one chunk with a 1536-dimensional embedding and the metadata key set above.

### Common failure

A `make load-data` that completes in under one second on a "cold" cache usually means the script picked up a pre-existing `data/chroma/` — `rm -rf data/chroma` is the bring-up step you skipped. Delete the directory and re-run.

## Exercise 2 — Measure recall@5 on a 12-question golden subset

Goal: run the filled embedder + store against twelve real questions from the golden set, report a per-question hit table, and discuss what the numbers mean for downstream evaluation.

### Why a 12-question subset

The full golden set is thirty questions across six query types: factual (12), procedural (6), conceptual (5), comparative (3), edge_case (2), and off_topic (2). For a retrieval-only measurement the last two buckets are misleading — edge-case questions ("Which classifier is the best one?") have ambiguous expected answers, and off-topic questions ("What's the weather in Paris today?") test the *refusal* pathway, not retrieval. They belong in the Module 03 (generator) and Module 20 (guardrails) evaluations, not here. The twelve-question subset below is five factual, three procedural, two conceptual, and two comparative — heavy on the buckets where a "right answer" is unambiguous enough to score.

### Steps

1. Write `scripts/recall_at_5.py` (create the file under the starter's `scripts/`):

   ```python
   """Recall@5 against a balanced subset of the golden set."""
   import csv, time
   from pathlib import Path
   from src import embedder, store

   GOLDEN = Path("data/golden_set.csv")
   SUBSET_SIZE = {"factual": 5, "procedural": 3, "conceptual": 2, "comparative": 2}

   def hit_any(returned_ids: list[str], expected_prefixes: list[str]) -> bool:
       return any(r.startswith(p) for r in returned_ids for p in expected_prefixes if p)

   def main() -> None:
       rows = list(csv.DictReader(GOLDEN.open()))
       remaining = dict(SUBSET_SIZE)
       picked = []
       for row in rows:
           if remaining.get(row["query_type"], 0) > 0:
               picked.append(row)
               remaining[row["query_type"]] -= 1
       start = time.monotonic()
       hits = 0
       for i, row in enumerate(picked, 1):
           qv = embedder.embed_query(row["question"])
           returned = [s.doc_id for s in store.query(qv, n_results=5)]
           expected = [p for p in row["expected_doc_ids"].split("|") if p]
           is_hit = hit_any(returned, expected)
           hits += is_hit
           print(f"Q{i:>2} [{row['query_type']:<11}] {'HIT ' if is_hit else 'MISS'} {row['question'][:70]}")
       elapsed = time.monotonic() - start
       print(f"\nrecall@5: {hits}/{len(picked)} = {hits/len(picked):.2f}")
       print(f"runtime:  {elapsed:.1f}s ({len(picked)} queries, ${len(picked)*2e-5:.5f})")

   if __name__ == "__main__":
       main()
   ```

   The `hit_any` helper implements the golden-set's documented "hit-any" semantics: a question scores 1 if any of the top-5 returned `doc_id`s starts with any of the pipe-separated expected prefixes. That matches the way `expected_doc_ids` is recorded as a section prefix (e.g., `modules.preprocessing`) rather than a specific chunk id, because the golden set was authored before chunking and should not be invalidated when a section splits into multiple pieces.

2. Run it from the starter root:

   ```
   uv run python scripts/recall_at_5.py
   ```

3. Compare your output to the reference run (twelve hits, eighteen seconds at warm-cache):

   ```
   Q 1 [factual    ] HIT  What is the default value of `n_estimators` in `RandomForestClassifier`?
   Q 2 [factual    ] HIT  What does `StandardScaler` do to input features?
   Q 3 [factual    ] HIT  What is the default value of `C` in `LogisticRegression`?
   Q 4 [factual    ] HIT  What distance metric does `KNeighborsClassifier` use by default?
   Q 5 [factual    ] HIT  What is the default `criterion` parameter for `DecisionTreeClassifier`?
   Q 6 [procedural ] HIT  How do I scale features before clustering them?
   Q 7 [procedural ] HIT  How do I split a dataset into a training and a test set?
   Q 8 [procedural ] HIT  How do I encode categorical string features for a linear model?
   Q 9 [conceptual ] HIT  Why does `StandardScaler` use `ddof=0` by default?
   Q10 [conceptual ] HIT  Why use stratified K-fold instead of plain K-fold for classification?
   Q11 [comparative] HIT  When should I use KMeans versus DBSCAN?
   Q12 [comparative] HIT  What's the difference between `OneHotEncoder` and `LabelEncoder`?

   recall@5: 12/12 = 1.00
   runtime:  17.9s (12 queries, $0.00024)
   ```

### Reading three of the questions in depth

Question 1 ("default value of `n_estimators` in `RandomForestClassifier`") is a textbook factual lookup. The expected sections — `modules.ensemble.random-forests`, `modules.ensemble.parameters`, `modules.ensemble.forests-of-randomized-trees` — together cover most of the scikit-learn ensemble user-guide page. The expected_doc_ids field is generous on purpose: the answer (100 trees, raised from 10 in version 0.22) lives in any of those sections depending on which scikit-learn version you read. A retriever that finds any of them within top-5 has done its job; choosing which one to cite is the generator's problem.

Question 6 ("How do I scale features before clustering them?") is procedural and crosses two user-guide pages — preprocessing and clustering. The expected_doc_ids set is `modules.preprocessing|modules.clustering` rather than a specific subsection, because the "correct" procedural answer is a two-step recipe that combines knowledge from both pages. This is the shape most procedural questions take in a docs corpus: the right answer is not in any single section, and the generator has to compose. Module 07 will show how the retrieval-plus-generation composition turns a multi-section hit into a coherent answer; Module 11 will show how to measure when the composition stays faithful to the cited sources versus drifting into model knowledge.

Question 12 ("difference between `OneHotEncoder` and `LabelEncoder`") is comparative. The expected_doc_ids set is `modules.preprocessing|modules.preprocessing_targets` because the answer needs both — `OneHotEncoder` lives in feature preprocessing, `LabelEncoder` lives in target preprocessing, and confusing the two is a real bug pattern in newer scikit-learn users. A retriever that returns only one of the two pages would still score as a hit under the lenient prefix rule, but the answer the generator produces from a one-sided hit would be incomplete. Top-K hit-any is the easiest retrieval metric to compute and the loosest one to argue from; the better evaluation question, which Module 11 builds out, is whether the retrieved set is jointly sufficient to answer the question.

### What the numbers mean

A perfect 12/12 is not a victory lap — it is the floor the smoke gate already pinned at recall@5 ≥ 0.70 on the full thirty-question set, and the smoke gate measured 1.00 there too. What you should take away is that on a corpus this small (~750 chunks) and with questions sourced from the scikit-learn docs themselves, recall is not the binding constraint on retrieval quality. The interesting failure modes show up at top-1 rank, not at top-5 hit-any.

To see one of those failure modes, modify the script to print the top-1 doc_id for each query and re-run. You will see entries like this for the factual questions:

```
Q 1 -> seeded.near_dup.random_forest_estimators_a  (hit)
Q 2 -> seeded.near_dup.standard_scaler_a           (hit)
Q 3 -> seeded.version_conflict.logreg_solver_old   (hit)
Q 4 -> seeded.version_conflict.knn_metric_new      (hit)
```

Those `seeded.*` ids are not from the scikit-learn corpus — they are the eight deliberately-confusing chunks the seed-difficulty step inserts. They rank top-1 above the real `modules.ensemble.*` and `modules.preprocessing.*` chunks for these short factual questions because they are tighter paraphrases of the question text itself, and a dense embedding model rewards textual paraphrase. Recall@5 still counts the hit (the real chunk is somewhere in the top-5), but the answer the generator builds in Module 07 will be assembled from the seeded chunk first. Module 11 walks this exact pattern — recall stays high, answer faithfulness collapses, and the gap is the reason RAGAS evaluates both metrics independently.

### Acceptance criterion

`scripts/recall_at_5.py` runs in under thirty seconds against a populated `scikit_docs` collection, prints a per-question hit line for twelve questions, and reports a recall@5 of at least 0.70 (the smoke-gate floor). Save the printed table — Exercise 3 will compare a sentence-transformers run against it.

## Exercise 3 — Swap to a local sentence-transformers embedder

Goal: replace `text-embedding-3-small` (1536-dim, hosted OpenAI) with `sentence-transformers/all-MiniLM-L6-v2` (384-dim, local), see what breaks, and decide whether the trade is worth it for this corpus.

The model swap is one line in `src/embedder.py`. The problem is that everything downstream — the chunks already in Chroma, the cosine-pinned HNSW index, the 1536-dimensional vectors in the embedding cache — was built against the OpenAI model. Mixing 1536-dim corpus vectors with 384-dim query vectors is the failure Chroma raises on first request.

### Steps

1. Write a side-by-side wrapper so you can run both embedders without modifying the production embedder. Create `scripts/embed_with_st.py`:

   ```python
   """Local sentence-transformers embedder for the Module 05 swap exercise."""
   from sentence_transformers import SentenceTransformer

   _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

   def embed_query_st(text: str) -> list[float]:
       vec = _model.encode(text, normalize_embeddings=True)
       return vec.tolist()
   ```

   `normalize_embeddings=True` is the sentence-transformers equivalent of OpenAI's built-in unit-normalization — without it, the cosine metric Chroma is configured with will produce distances outside [0, 2].

2. Install the dependency:

   ```
   uv add sentence-transformers
   ```

   The first import downloads the ~80 MB MiniLM checkpoint to `~/.cache/huggingface/`. Allow about 60 seconds on a cold cache; subsequent imports are instant.

3. Try a query without rebuilding the collection (this is the failure to see):

   ```
   uv run python -c "
   from src import store
   from scripts.embed_with_st import embed_query_st
   qv = embed_query_st('How do I tune hyperparameters with grid search?')
   print(f'query vec dim: {len(qv)}')
   store.query(qv, n_results=3)
   "
   ```

   You should get a `chromadb.errors.InvalidDimensionException` telling you the collection holds 1536-dim vectors and your query is 384-dim. The collection's vector dimension is fixed at first insertion; there is no in-place re-embedding. You have to rebuild.

4. Rebuild into a *separate* collection so the OpenAI collection stays intact for comparison. Create `scripts/load_data_st.py`:

   ```python
   """MiniLM rebuild into a parallel collection for Exercise 3."""
   import time
   from pathlib import Path
   from src import store
   from src.chunker import chunk_doc
   from src.corpus import load_corpus
   from scripts.embed_with_st import _model

   COLLECTION = "scikit_docs_st"

   def main():
       start = time.monotonic()
       chunks = [c for sec in load_corpus(Path("data/scikit-learn-cache"), "manual")
                 for c in chunk_doc(sec)]
       texts = [c["text"] for c in chunks]
       embeddings = _model.encode(
           texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True
       ).tolist()
       col = store.get_collection(COLLECTION)
       col.upsert(
           ids=[c["chunk_id"] for c in chunks],
           documents=texts,
           embeddings=embeddings,
           metadatas=[{"doc_id": c["chunk_id"]} for c in chunks],
       )
       print(f"upserted {len(chunks)} into '{COLLECTION}' in {time.monotonic()-start:.1f}s")

   if __name__ == "__main__":
       main()
   ```

   Run it: `uv run python scripts/load_data_st.py`. On a CPU-only Workspace, expect ~90 seconds for ~750 chunks — slower than the batched OpenAI path because the local model runs serially without an accelerator, but the throughput is acceptable for a corpus this small. Cost: zero.

5. Re-run the recall measurement against the new collection. Copy `scripts/recall_at_5.py` to `scripts/recall_at_5_st.py` and change two things: import `embed_query_st` in place of `embedder.embed_query`, and query the `scikit_docs_st` collection directly (`store.get_collection("scikit_docs_st").query(...)`), converting distances to similarity yourself the same way the public `store.query` does.

6. Compare. A representative side-by-side on the twelve-question subset:

   | Embedder | dim | recall@5 | corpus-build cost | query latency (median) |
   |---|---|---|---|---|
   | `text-embedding-3-small` | 1536 | 12/12 = 1.00 | ~$0.10 (cold) / $0 (warm cache) | ~1.5s (network) |
   | `all-MiniLM-L6-v2` | 384 | 10/12 = 0.83 | $0 (local) | ~0.05s (in-process) |

   Two questions that hit under OpenAI but missed under MiniLM are typically the ones whose expected sections are short and lexically distant from the question wording — MiniLM's 384-dim space packs less surface-form variation into the same neighborhood, so paraphrase recall drops noticeably even when the topic is correct. The flip side is that query latency drops by an order of magnitude and the cost goes to zero.

### Acceptance criterion

You can produce a side-by-side comparison table like the one above, with your own numbers, for both embedders on the twelve-question subset. You can explain, in two sentences, the reason the dimensions do not match across collections and why the rebuild is unavoidable.

### Why the dimensionalities cannot be reconciled in place

Chroma's HNSW index is a graph data structure where every vector in the collection is a node, and edges connect nearby vectors. The "nearby" relation is computed by the distance function declared at create time — here, cosine — and the function operates on vectors of a fixed dimension. The dimension is determined by the first vector inserted; subsequent inserts at a different dimension would either be silently truncated or padded with zeros, which would corrupt the index, so Chroma refuses them with `InvalidDimensionException` and forces you to surface the mismatch.

This is not a design limitation specific to Chroma. Every dense vector store works this way: pgvector with an `IVFFlat` index, Pinecone with its serverless pods, Weaviate with its HNSW segments — all of them fix dimensionality at first insert and reject mismatched queries. The only way to "swap embedders" in a production system is to build a parallel collection the way Exercise 3 does, then atomically swap which collection the application reads from. Module 24 (RAGOps) walks the blue/green pattern end-to-end and reuses Exercise 3's two-collection setup as its starting point.

### What stays the same across embedders

Two pieces of the Module 05 stack survive the embedder swap untouched: `src/chunker.py` and the structure of `src/store.py`. Chunking is a function of the source corpus, not the embedder, so the same chunk_doc output feeds both rebuilds. The store's `get_collection / add / query` API is also embedder-agnostic — the only thing that varies is which collection name you pass in and what dimensionality the vectors happen to have. That separation is why the rebuild script is so short: most of the work is already done.

### When the trade is worth it

For the ScikitDocs corpus, the OpenAI embedder is the right default — 1.00 versus 0.83 on twelve questions is a meaningful gap and the cost is rounding-error per build because the embedding cache makes warm rebuilds free. The MiniLM swap matters when you have a much larger corpus where the per-rebuild cost is no longer rounding-error, when you cannot send the corpus to a hosted API (regulated industries, air-gapped deployments), or when query latency under 50 milliseconds is a hard product constraint. Pick the embedder against the constraint that is binding for the workload — measure on your own queries the way the MTEB paper recommends, do not pick from a public leaderboard.

A second axis to plan for, even if you stay on OpenAI: model upgrades. OpenAI shipped `text-embedding-3-large` and the Matryoshka-truncatable `text-embedding-3-small` together; the latter is what we use. The two are not interchangeable in an existing collection (different dim, different geometry), so a model upgrade follows the same blue/green rebuild pattern as an embedder swap. Plan for it now by keeping the embedder configuration in `src/config.py` and the rebuild script generic, both of which Exercise 3 already exercises.
