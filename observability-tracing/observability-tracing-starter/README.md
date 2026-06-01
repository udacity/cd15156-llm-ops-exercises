# ScikitDocs starter

Unified alt-workload starter for the 11 implementation modules of
Udacity nd907 Course 2 (LLM Ops). ScikitDocs is a Q&A assistant for the
[scikit-learn](https://scikit-learn.org) library — learners build it
incrementally across Module 03 through Module 26 by filling in stubbed `src/` files,
then prove their LLM Ops skills on the unrelated capstone (ThirdShotHub
pickleball FAQ) at `project/`.

## Why ScikitDocs

The capstone is a *project* — it must remain a clean slate for learners.
Walking through it module-by-module during instruction would make the
capstone feel like a re-walk rather than an application of skills.
ScikitDocs gives instruction a permanent, well-known, freely-available
documentation corpus to teach against. The capstone stays an exercise
in *applying* what was taught.

## Architecture — Option C′ (stubbed shared `src/`)

`src/` is a type-stubbed skeleton at scaffold time. Each implementation
module fills in the file it teaches:

- **Module 03 (Prompt Versioning)** → fills `src/generator.py`
- **Module 05 (Vector DB)** → fills `src/store.py`, `src/embedder.py`, `src/chunker.py`
- **Module 07 (RAG Pipeline)** → fills `src/pipeline.py`
- **Module 09 (Phoenix Tracing)** → instruments `src/pipeline.py` + `src/gateway/`
- **Module 11 (RAGAS Evaluation)** → builds `scripts/run_eval.py`
- **Module 13 (Cost Monitoring)** → adds `src/pricing.py` + cost tracking
- **Module 15 (Semantic Cache)** → adds `src/cache/`
- **Module 18 (Gateway)** → adds `src/gateway/` with FastAPI app + `X-Client-Id` header
- **Module 20 (Guardrails)** → adds `src/guardrails/`
- **Module 22 (A/B Testing)** → adds sticky-by-user assignment via `client_id`
- **Module 24 (RAGOps)** → adds blue/green index swap
- **Module 26 (Latency)** → adds a streaming variant of `/query`

`exercises/` is flat per module — each module's hands-on work lives in
its own `exercises/m{NN}_<slug>/` subdirectory. Exercises READ from
`src/` but never WRITE to it, which preserves commit-and-/clear
discipline between modules.

`src/constants.py` is the **authoritative source** for 21 cross-module
invariants (port, models, top-k, thresholds, route paths, header
casing). Read [`CONSTANTS.md`](CONSTANTS.md) for the rationale.
[`INTERFACES.md`](INTERFACES.md) freezes the function signatures.

## Mapping to the capstone

| ScikitDocs path | Capstone counterpart (`project/`) | Purpose |
|---|---|---|
| `src/store.py` | `project/src/vectordb/store.py` | Chroma client + collection management |
| `src/embedder.py` | `project/src/vectordb/embedder.py` | OpenAI embeddings wrapper |
| `src/chunker.py` | `project/src/vectordb/chunker.py` | Document chunking |
| `src/generator.py` | `project/src/rag/generator.py` | OpenAI chat + Jinja2 prompt rendering |
| `src/pipeline.py` | `project/src/rag/pipeline.py` | End-to-end RAG composition |
| `src/models.py` | `project/src/models.py` | `Source`, `TokenUsage`, `QueryResponse` |
| `src/config.py` | `project/src/config.py` | Settings via pydantic-settings + `.env` |
| `src/pricing.py` | `project/src/pricing.py` | Per-model cost table |
| `src/gateway/` | `project/src/gateway/` | FastAPI app + routes |
| `src/guardrails/` | `project/src/guardrails/` | Input/output safety |
| `src/cache/` | `project/src/cache/` | Semantic answer cache |
| `prompts/docbot_system.j2` | `project/prompts/rag_system.j2` | Domain-specific system prompt |
| `scripts/load_data.py` | `project/scripts/load_data.py` | Corpus ingestion |
| `scripts/run_eval.py` | `project/src/evaluation/run_eval.py` | RAGAS evaluation |
| `data/golden_set.csv` | `project/data/golden_set.csv` | RAGAS golden Q&A |
| `Makefile` | `project/Makefile` | `setup`, `serve`, `load-data`, `eval`, `test`, `verify` |

The shape is intentionally lighter than the capstone — fewer packages,
simpler routing — so the focus stays on the LLM Ops concept of each
module rather than incidental complexity.

## Setup

```bash
uv sync                       # installs deps into .venv/
cp .env.example .env          # add your OPENAI_API_KEY (or Vocareum voc- key)
make test                     # smoke test passes immediately at scaffold time
```

Each implementation module's exercise extends the starter incrementally.
Run `make serve` after the initial scaffolding (Module 18 gateway) lands. Run `make eval`
after the initial scaffolding (Module 11 evaluation) lands. See each module's `code-refs.md`
for the prerequisite REQs.

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

## Provenance

- Plan: [`docs/plans/2026-05-17-feat-rewrite-impl-modules-scikitdocs-altworkload-plan.md`](../../docs/plans/2026-05-17-feat-rewrite-impl-modules-scikitdocs-altworkload-plan.md)
- Scaffold REQ: the initial scaffolding plan (post-archive path)
- Capstone constraint: zero edits to `project/` from any Wave 4 REQ. Verified at the initial scaffolding.
