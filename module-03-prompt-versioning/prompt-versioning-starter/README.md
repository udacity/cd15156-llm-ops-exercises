# ScikitDocs starter

ScikitDocs is a Q&A assistant for the
[scikit-learn](https://scikit-learn.org) library. You build it
incrementally across the implementation modules by filling in stubbed
`src/` files — each module implements the piece it teaches against this
shared codebase. It gives you a permanent, well-known, freely-available
documentation corpus to practice LLM Ops skills against.

## Layout

`src/` is a type-stubbed skeleton. Each module's exercise fills in the
file it teaches, leaving the rest of the contract intact.

`exercises/` holds each module's hands-on work in its own subdirectory.
Exercises read from `src/` but never write to it.

`src/constants.py` is the authoritative source for the shared constants
(ports, models, top-k, thresholds, route paths). Read
[`CONSTANTS.md`](CONSTANTS.md) for the values and
[`INTERFACES.md`](INTERFACES.md) for the frozen function signatures.

## Setup

```bash
uv sync                       # installs deps into .venv/
cp .env.example .env          # add your OPENAI_API_KEY (or Vocareum voc- key)
make test                     # smoke test passes immediately at scaffold time
```

Each module's exercise extends the starter incrementally. Other Makefile
targets (`make serve`, `make eval`) become available once you implement
the pieces they depend on.

### Populating the corpus (`make load-data`)

`make load-data` shallow-clones scikit-learn at the pinned tag
(`corpus.SCIKIT_LEARN_TAG` — currently `1.5.2`), parses every RST under
`doc/modules/`, `doc/tutorial/`, and `doc/auto_examples/` (if present)
via docutils with the stub-role pattern, splits long sections into
~450-token paragraph-bounded chunks, and upserts them into a Chroma
collection (`scikit_docs`, `hnsw:space=cosine`).

Build artefacts (all gitignored):

- `data/scikit-learn-cache/` — the shallow clone (~250 MB)
- `data/embedding_cache.jsonl` — embeddings keyed by SHA-256 of chunk
  text; reused across iterations so a re-run hits Chroma in seconds
- `data/CORPUS_VERSION` — `tag`, `sha`, `ingest_timestamp`, `chunk_count`
- `data/chroma/` — the persistent Chroma store

Target wall-clock on the Udacity Workspace (4 GB RAM / 2 vCPU):

- **Cold** (no caches): ~45–60 seconds — dominated by the OpenAI
  embedding API; 16 batches of 256 chunks ride four parallel requests.
- **Warm** (caches present): under five seconds — embeddings replay
  from `embedding_cache.jsonl`, Chroma idempotently upserts.

Cost per cold build is ~$0.08–0.15 against `text-embedding-3-small`.

### Resetting (`make clean`)

`make clean` deletes the generated artefacts above — the venv plus every
corpus cache (`data/chroma/`, `data/scikit-learn-cache/`,
`data/embedding_cache.jsonl`, `data/CORPUS_VERSION`, `data/phoenix/`). Your
source data (golden set, seeded chunks) is tracked in Git and left untouched.

**Cost impact of a reset.** `make setup` after a clean is free — it only
reinstalls Python packages, no API calls. The charge lands on the next
`make load-data`: because clean removed `embedding_cache.jsonl`, that run is a
**cold** rebuild that re-embeds the whole corpus (~$0.08–0.15, ~45–60 s)
instead of a sub-five-second warm replay. So each full
`clean` → `setup` → `load-data` cycle costs about one cold build.
