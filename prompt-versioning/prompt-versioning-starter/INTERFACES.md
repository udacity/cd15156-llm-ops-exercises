# Frozen interfaces — ScikitDocs starter

These signatures are locked at REQ-061. Each impl REQ fills in the body
of the file its module teaches. Changing a signature requires the
upstream-patchback or infra-amendment protocols documented in
`docs/plans/2026-05-17-feat-rewrite-impl-modules-scikitdocs-altworkload-plan.md`
§"Operational protocols".

When you fill in a stub, replace `raise NotImplementedError(...)` with the
real implementation but **do not change the signature, return type, or
docstring contract**. The downstream modules depend on this shape.

## Frozen function contracts

### `src/corpus.py` — filled by REQ-062 (Wave 4 infra)

```python
def load_corpus(repo_path: Path, version_sha: str) -> Iterator[dict]
```

Walks a pinned scikit-learn checkout, parses `doc/**/*.rst` with
docutils, yields one dict per top-level section.

**Yielded dict shape:**
- `text` — section body, RST stripped to plain text
- `metadata` — `{source_path, section_title, has_code, code_languages, xrefs}`
- `doc_id` — stable id derived from path + section anchor

### `src/chunker.py` — filled by REQ-065 (M05)

```python
def chunk_doc(doc: dict) -> list[dict]
```

Section-header chunking. Target ~500 tokens
(`constants.CHUNK_TARGET_TOKENS`), overlap ~75 tokens
(`constants.CHUNK_OVERLAP_TOKENS`). Preserves `has_code`,
`code_languages`, `xrefs` metadata from the source doc.

Each returned dict has `text`, `metadata`, `chunk_id`.

### `src/embedder.py` — filled by REQ-065 (M05)

```python
def embed(text: str | list[str]) -> list[float] | list[list[float]]
def embed_query(text: str) -> list[float]
```

`embed` MUST send a single batched OpenAI request when given a list. The
batched-load path (REQ-062 `scripts/load_data.py`) targets ≥256 chunks per
request.

### `src/store.py` — filled by REQ-065 (M05)

```python
def get_collection(name: str = "scikit_docs") -> Collection
def add(documents, embeddings, metadatas, ids) -> None
def query(query_embedding: list[float], n_results: int = 5) -> list[Source]
```

Wraps `chromadb.PersistentClient`. Default collection name is
`scikit_docs`. `query` returns `Source` instances sorted by similarity
score (highest first).

### `src/generator.py` — filled by REQ-064 (M03)

```python
def render_system_prompt(sources: list[Source]) -> str
def generate(question: str, sources: list[Source], model: str) -> tuple[str, TokenUsage, float]
```

`render_system_prompt` reads `prompts/docbot_system.j2` and renders it
with the retrieved chunks. `generate` calls OpenAI chat completions and
returns the answer string, token usage, and computed cost.

### `src/pipeline.py` — filled by REQ-066 (M07)

```python
def run_pipeline(question: str, top_k: int = 5, model: str | None = None) -> QueryResponse
```

End-to-end RAG composition: `embed_query` → `store.query` →
`generator.generate`. Returns a `QueryResponse` (see `src/models.py`).
`model=None` defaults to `settings.model_complex`.

## Module-fills-stub matrix

| File | Filled by | Module |
|---|---|---|
| `src/corpus.py` | REQ-062 | (infra) |
| `src/chunker.py` | REQ-065 | M05 |
| `src/embedder.py` | REQ-065 | M05 |
| `src/store.py` | REQ-065 | M05 |
| `src/generator.py` | REQ-064 | M03 |
| `src/pipeline.py` | REQ-066 | M07 |
| `src/pricing.py` (added) | REQ-069 | M13 |
| `src/cache/` (added) | REQ-070 | M15 |
| `src/gateway/` (added) | REQ-071 | M18 |
| `src/guardrails/` (added) | REQ-072 | M20 |
| `src/optimization/` (added) | REQ-073 | M22 |
| `src/ingestion/` (added) | REQ-074 | M24 |
| `src/streaming/` (added) | REQ-075 | M26 |
| Phoenix tracing instrumentation across `src/pipeline.py` + `src/gateway/` | REQ-067 | M09 |

## Cross-module contracts

### `X-Client-Id` header contract (REQ-071 → REQ-073)

The gateway accepts an optional `X-Client-Id` request header
(`constants.CLIENT_ID_HEADER`). When present, REQ-073 (M22 A/B
testing) uses the header value as the bucketing key for
sticky-by-user variant assignment. A contract test in
`tests/test_smoke.py` (added by REQ-071) verifies the header is
plumbed end-to-end before M22 lands.

### Collection naming (REQ-074)

M24 RAGOps demonstrates blue/green index swap by creating two
collections (`scikit_docs_blue`, `scikit_docs_green`) and updating
which one `get_collection()` returns. Default name `scikit_docs`
is the alias.

### Streaming endpoint (REQ-075)

M26 latency adds a streaming variant of `/query` to demonstrate
time-to-first-token improvements. The non-streaming `/query` route
stays unchanged so M11 evaluation (REQ-068) and other downstream
consumers aren't broken.
