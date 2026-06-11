> A walkthrough of the codebase you'll work with. See INSTRUCTIONS.md for the exercise tasks.

# Demo: Build the ScikitDocs Vector-DB Layer From Three Stubs

The concept module argued that an embedding store is three things stacked: a chunker, an embedder, and an indexed key/value backend. This demo builds those three pieces in the ScikitDocs starter from frozen stubs — `src/chunker.py`, `src/embedder.py`, `src/store.py` — runs them end to end against the scikit-learn 1.5.2 docs corpus, and shows the one piece of Chroma configuration (`hnsw:space=cosine`) that quietly determines whether your similarity ranking is meaningful at all.

## Setup

You should already have this starter cloned and `make setup` complete. If not, run that first; the rest of the demo will not work end-to-end without it. The demo assumes:

- This starter directory is your working directory for every command below.
- `.env` has `OPENAI_API_KEY` set. If you are on Vocareum, the key starts with `voc-` and `.env` also has `OPENAI_BASE_URL=https://openai.vocareum.com/v1`. If you are on direct OpenAI, leave `OPENAI_BASE_URL` empty. The starter reads both through `src/config.py` and forwards them to the OpenAI client; the same code path works in either environment.
- The corpus is loaded: `make load-data`. The first run takes ~30 seconds — it shallow-clones scikit-learn at tag `1.5.2`, parses every RST file under `doc/modules/` and `doc/tutorial/`, chunks at section boundaries, embeds in 256-chunk batches with four parallel in-flight requests, and upserts into Chroma's `scikit_docs` collection. Subsequent runs reuse the embedding cache at `data/embedding_cache.jsonl` and complete in under five seconds.

Sanity check from the starter root:

```
uv run python -c "from src import store; print(store.get_collection().count())"
```

You want a number around 750 — that is the chunk count after `make load-data` and `make seed-difficulty` have both run. Zero means the corpus is empty (run `make load-data`); an import error means `src/store.py` is still a stub (Part 3 fills it in).

## Part 1 — Build `chunker.py` from the stub

Open `src/chunker.py`. It started as a stub — one function, `chunk_doc`, raising `NotImplementedError`, with its signature frozen by `INTERFACES.md`. Filled in, it is the piece every chunk that enters the vector store passes through. Walk through the file from top to bottom.

The strategy is section-header chunking. The input is one dict per top-level section, already yielded by `src/corpus.py` — that file is the corpus parser's work and we leave it alone. Each input dict has `text` (the section body, stripped of RST markup), `doc_id` (a stable id like `modules.cross_validation.grid-search`), and a `metadata` block with `source_path`, `section_title`, `section_path`, `url`, `has_code`, `code_languages`, `xrefs`, and `scikit_learn_sha`. The chunker decides whether that section is one chunk or several.

The decision is two tiers:

```python
if tokens < _CHUNK_MIN_TOKENS:
    return []
if tokens <= _CHUNK_MAX_TOKENS:
    return [{"text": text, "metadata": base_metadata, "chunk_id": base_id}]
pieces = _split_long_text(text, _CHUNK_MAX_TOKENS, constants.CHUNK_OVERLAP_TOKENS)
```

Sections under 50 tokens are dropped — they are noise for retrieval (one-line "See also" stubs, residual headings). Sections at or under 512 tokens pass through as a single chunk with the original `doc_id`. Only sections above 512 tokens go through `_split_long_text`. That function walks paragraphs and packs them into 512-token windows with 75-token overlap; if a single paragraph is already over the ceiling it is emitted as-is rather than mid-paragraph-split, because mid-paragraph splits hurt embedding semantics more than slightly oversized chunks hurt recall.

Three details worth naming. First, `has_code`, `code_languages`, and `xrefs` propagate at the section level — every chunk from a section inherits the parent's flags, even if a particular split piece does not contain the code block itself. Code-aware retrieval analysis downstream relies on `has_code`, and the section is the right granularity for that flag. Second, `chunk_id` differs from `doc_id` for split sections: the first piece gets `<doc_id>.p0`, the second `<doc_id>.p1`, and so on. That naming is load-bearing — `src/store.py` uses `chunk_id` as the Chroma primary key, so the postfix is how re-runs stay idempotent without colliding multi-piece sections. Third, `corpus.token_count` is the project-wide encoder (`cl100k_base`, the encoding for both `text-embedding-3-small` and `gpt-4o`), so token counts here match what OpenAI bills downstream.

Try one section now:

```
uv run python -c "
from pathlib import Path
from src import chunker, corpus
sec = next(s for s in corpus.load_corpus(Path('data/scikit-learn-cache'), 'demo')
           if 'grid_search' in s['doc_id'])
chunks = chunker.chunk_doc(sec)
print(f'section: {sec[\"doc_id\"]} ({corpus.token_count(sec[\"text\"])} tokens)')
print(f'-> {len(chunks)} chunks, first chunk_id={chunks[0][\"chunk_id\"]}, has_code={chunks[0][\"metadata\"][\"has_code\"]}')
"
```

You should see a multi-chunk split if the section is over 512 tokens, with `has_code=True` propagating to every piece because the section embeds Python code examples.

## Part 2 — Build `embedder.py` from the stub

Open `src/embedder.py`. The contract from `INTERFACES.md` is two functions — `embed(text: str | list[str])` and `embed_query(text: str)` — and one performance requirement: list inputs MUST be sent in batched OpenAI requests of at least 256 inputs per request.

The why for that minimum is mechanical, not stylistic. Embedding 4,000 chunks one at a time hits the OpenAI API 4,000 times, each call paying its own TLS handshake, request overhead, and rate-limit token-bucket round-trip. On a warm connection from the Vocareum Workspace, that takes roughly thirteen minutes. The same 4,000 chunks batched at 256 inputs per call hit the API sixteen times and finish in about thirty seconds. The batched path is twenty-five times faster for the same number of tokens billed — the difference is entirely request overhead, and 256 is the elbow on the cost-per-chunk curve where added inputs per request stop reducing overhead in a meaningful way.

The filled function does exactly that:

```python
def embed(text: str | list[str]) -> list[float] | list[list[float]]:
    if isinstance(text, str):
        return _embed_batch([text])[0]
    if not text:
        return []
    vectors: list[list[float]] = []
    for start in range(0, len(text), _BATCH_SIZE):
        vectors.extend(_embed_batch(text[start : start + _BATCH_SIZE]))
    return vectors
```

Single string in, single vector out. List of strings in, one OpenAI request per 256-input slice, vectors concatenated in input order. `_embed_batch` is the tight loop: one call to `client.embeddings.create(input=texts, model=settings.embedding_model)`, returning `[item.embedding for item in response.data]`. `embed_query` is a thin alias — same call path, always a single input — kept as its own function so query-time call sites read naturally and so a query-side cache can later be swapped in without touching the corpus-side path.

Two configuration choices that route through `src/config.py` and `src/constants.py`. The model name comes from `settings.embedding_model`, which defaults to `constants.EMBEDDING_MODEL = "text-embedding-3-small"` (1536-dim, unit-normalized). The OpenAI client is built via `_client()`, lazily, with `api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY")` and `base_url = settings.openai_base_url or None`. The empty-string-to-None fall-through is the Vocareum bridge: an empty `OPENAI_BASE_URL` means the SDK uses its built-in default; the Vocareum URL routes through the proxy. Same code, two deploy targets, zero conditional branches.

You already ran `make load-data` in Setup. The end of its output prints the embedding throughput — look back at it now and confirm the "embedded N new + reused M cached in <30s" line for a cold build lines up with the thirty-second number quoted above. A warm build, where `embedding_cache.jsonl` already has every chunk, finishes in under five seconds because no API calls fire at all.

## Part 3 — Build `store.py` from the stub

Open `src/store.py`. Three functions wrap `chromadb.PersistentClient`: `get_collection`, `add`, `query`. The whole file is sixty lines; the one line that matters most is in `get_collection`:

```python
return _client().get_or_create_collection(
    name=name,
    metadata={"hnsw:space": "cosine"},
)
```

Chroma's default distance metric is L2 (squared Euclidean). OpenAI's `text-embedding-3-small` vectors are unit-normalized, which means cosine similarity and the negative inner product are equivalent up to a sign, and L2 distance and cosine distance are monotonically related — but only up to a constant. Ranking under L2 versus cosine on normalized vectors gives the same top-K in theory, but Chroma's HNSW index parameters are tuned for the metric you declare at create time, and the "1 minus distance" conversion only produces a meaningful similarity score under cosine. Forget the metadata kwarg, and your retrieval still returns vectors, but `similarity_score` is now a number with no useful interpretation. The pin is load-bearing.

`add` uses `upsert`, not `add`:

```python
get_collection().upsert(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
```

Idempotent on `ids`. Re-running `make load-data` against the same corpus does not raise on duplicates; it overwrites in place. The same property is what lets a blue/green index swap rebuild a collection without dropping the live one.

`query` translates Chroma's cosine *distance* (where smaller is better) into a similarity *score* (where larger is better) on the way out:

```python
sources = [
    Source(doc_id=doc_id, chunk_text=document, similarity_score=1.0 - distance)
    for doc_id, document, distance in zip(ids, documents, distances)
]
sources.sort(key=lambda s: s.similarity_score, reverse=True)
```

Callers see `list[Source]` sorted highest first. The `Source` model is `src/models.py` — three fields, `doc_id`, `chunk_text`, `similarity_score`.

End-to-end check, all three pieces composed:

```
uv run python -c "
from src import embedder, store
qv = embedder.embed_query('How do I tune hyperparameters with grid search?')
for s in store.query(qv, n_results=3):
    print(f'{s.similarity_score:.3f}  {s.doc_id}')
"
```

Top-1 should be in `modules.grid_search.*` with a similarity of roughly 0.58. The score is not "58 percent confidence" — it is the cosine similarity of two unit vectors in a 1536-dimensional space, where 1.0 is identical and 0.0 is orthogonal — and chunks above ~0.5 are typically on-topic for short factual questions. The confidence-vs-similarity distinction matters later once you start logging it.

## Wrap-up

That is the full vector-DB layer, end to end. A chunker that respects section boundaries and propagates `has_code` metadata. An embedder that batches 256 inputs per OpenAI request. A store that pins cosine, upserts idempotently, and converts distance to a usable similarity score. The pieces are small and the indirection is shallow, which is the strength of this layer — no orchestration framework, no abstract base classes, three files of about sixty lines each.

The exercises take this further: standing up the collection from scratch and verifying its contents, measuring recall@5 against a twelve-question subset of the golden set so you have a real number to argue from, and swapping `text-embedding-3-small` for `sentence-transformers/all-MiniLM-L6-v2` to feel what a dimensional mismatch looks like at query time and what the trade-off curve between hosted and local embedders actually buys.

---

