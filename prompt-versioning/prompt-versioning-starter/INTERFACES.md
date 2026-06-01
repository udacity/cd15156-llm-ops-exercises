# Frozen interfaces — ScikitDocs starter

These signatures are locked at the initial scaffolding. Each impl REQ fills in the body
of the file its module teaches. Changing a signature requires the
upstream-patchback or infra-amendment protocols documented in
`docs/plans/2026-05-17-feat-rewrite-impl-modules-scikitdocs-altworkload-plan.md`
§"Operational protocols".

When you fill in a stub, replace `raise NotImplementedError(...)` with the
real implementation but **do not change the signature, return type, or
docstring contract**. The downstream modules depend on this shape.

## Frozen function contracts

### `src/corpus.py` — filled by the initial scaffolding (Wave 4 infra)

```python
def load_corpus(repo_path: Path, version_sha: str) -> Iterator[dict]
```

Walks a pinned scikit-learn checkout, parses `doc/**/*.rst` with
docutils, yields one dict per top-level section.

**Yielded dict shape:**
- `text` — section body, RST stripped to plain text
- `metadata` — `{source_path, section_title, has_code, code_languages, xrefs}`
- `doc_id` — stable id derived from path + section anchor

### `src/chunker.py` — filled by Module 05

```python
def chunk_doc(doc: dict) -> list[dict]
```

Section-header chunking. Target ~500 tokens
(`constants.CHUNK_TARGET_TOKENS`), overlap ~75 tokens
(`constants.CHUNK_OVERLAP_TOKENS`). Preserves `has_code`,
`code_languages`, `xrefs` metadata from the source doc.

Each returned dict has `text`, `metadata`, `chunk_id`.

### `src/embedder.py` — filled by Module 05

```python
def embed(text: str | list[str]) -> list[float] | list[list[float]]
def embed_query(text: str) -> list[float]
```

`embed` MUST send a single batched OpenAI request when given a list. The
batched-load path (the initial scaffolding `scripts/load_data.py`) targets ≥256 chunks per
request.

### `src/store.py` — filled by Module 05

```python
def get_collection(name: str = "scikit_docs") -> Collection
def add(documents, embeddings, metadatas, ids) -> None
def query(query_embedding: list[float], n_results: int = 5) -> list[Source]
```

Wraps `chromadb.PersistentClient`. Default collection name is
`scikit_docs`. `query` returns `Source` instances sorted by similarity
score (highest first).

### `src/generator.py` — filled by Module 03

```python
def render_system_prompt(sources: list[Source]) -> str
def generate(question: str, sources: list[Source], model: str) -> tuple[str, TokenUsage, float]
```

`render_system_prompt` reads `prompts/docbot_system.j2` and renders it
with the retrieved chunks. `generate` calls OpenAI chat completions and
returns the answer string, token usage, and computed cost.

### `src/pipeline.py` — filled by Module 07

```python
def run_pipeline(question: str, top_k: int = 5, model: str | None = None) -> QueryResponse
```

End-to-end RAG composition: `embed_query` → `store.query` →
`generator.generate`. Returns a `QueryResponse` (see `src/models.py`).
`model=None` defaults to `settings.model_complex`.

## Module-fills-stub matrix

| File | Filled by | Module |
|---|---|---|
| `src/corpus.py` | the initial scaffolding | (infra) |
| `src/chunker.py` | Module 05 |
| `src/embedder.py` | Module 05 |
| `src/store.py` | Module 05 |
| `src/generator.py` | Module 03 |
| `src/pipeline.py` | Module 07 |
| `src/pricing.py` (added) | Module 13 |
| `src/cache/` (added) | Module 15 |
| `src/gateway/` (added) | Module 18 |
| `src/guardrails/` (added) | Module 20 |
| `src/optimization/` (added) | Module 22 |
| `src/ingestion/` (added) | Module 24 |
| `src/streaming/` (added) | Module 26 |
| Phoenix tracing instrumentation across `src/pipeline.py` + `src/gateway/` | Module 09 |

## Cross-module contracts

### `X-Client-Id` header contract (the initial scaffolding → the initial scaffolding)

The gateway accepts an optional `X-Client-Id` request header
(`constants.CLIENT_ID_HEADER`). When present, the initial scaffolding (Module 22 A/B
testing) uses the header value as the bucketing key for
sticky-by-user variant assignment. A contract test in
`tests/test_smoke.py` (added by the initial scaffolding) verifies the header is
plumbed end-to-end before Module 22 lands.

### Collection naming

Module 24 RAGOps demonstrates blue/green index swap by creating two
collections (`scikit_docs_blue`, `scikit_docs_green`) and updating
which one `get_collection()` returns. Default name `scikit_docs`
is the alias.

### Streaming endpoint

Module 26 latency adds a streaming variant of `/query` to demonstrate
time-to-first-token improvements. The non-streaming `/query` route
stays unchanged so Module 11 evaluation and other downstream
consumers aren't broken.
