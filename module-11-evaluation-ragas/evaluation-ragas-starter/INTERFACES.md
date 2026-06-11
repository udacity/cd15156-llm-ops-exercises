# Frozen interfaces — ScikitDocs starter

These signatures are frozen. Implement the body of the file your exercise
teaches, but **do not change the signature, return type, or docstring
contract** — other parts of the starter depend on this shape.

When you fill in a stub, replace `raise NotImplementedError(...)` with the
real implementation.

## Frozen function contracts

### `src/corpus.py`

```python
def load_corpus(repo_path: Path, version_sha: str) -> Iterator[dict]
```

Walks a pinned scikit-learn checkout, parses `doc/**/*.rst` with
docutils, yields one dict per top-level section.

**Yielded dict shape:**
- `text` — section body, RST stripped to plain text
- `metadata` — `{source_path, section_title, has_code, code_languages, xrefs}`
- `doc_id` — stable id derived from path + section anchor

### `src/chunker.py`

```python
def chunk_doc(doc: dict) -> list[dict]
```

Section-header chunking. Target ~500 tokens
(`constants.CHUNK_TARGET_TOKENS`), overlap ~75 tokens
(`constants.CHUNK_OVERLAP_TOKENS`). Preserves `has_code`,
`code_languages`, `xrefs` metadata from the source doc.

Each returned dict has `text`, `metadata`, `chunk_id`.

### `src/embedder.py`

```python
def embed(text: str | list[str]) -> list[float] | list[list[float]]
def embed_query(text: str) -> list[float]
```

`embed` MUST send a single batched OpenAI request when given a list. The
batched-load path (`scripts/load_data.py`) targets ≥256 chunks per
request.

### `src/store.py`

```python
def get_collection(name: str = "scikit_docs") -> Collection
def add(documents, embeddings, metadatas, ids) -> None
def query(query_embedding: list[float], n_results: int = 5) -> list[Source]
```

Wraps `chromadb.PersistentClient`. Default collection name is
`scikit_docs`. `query` returns `Source` instances sorted by similarity
score (highest first).

### `src/generator.py`

```python
def render_system_prompt(sources: list[Source]) -> str
def generate(question: str, sources: list[Source], model: str) -> tuple[str, TokenUsage, float]
```

`render_system_prompt` reads `prompts/docbot_system.j2` and renders it
with the retrieved chunks. `generate` calls OpenAI chat completions and
returns the answer string, token usage, and computed cost.

### `src/pipeline.py`

```python
def run_pipeline(question: str, top_k: int = 5, model: str | None = None) -> QueryResponse
```

End-to-end RAG composition: `embed_query` → `store.query` →
`generator.generate`. Returns a `QueryResponse` (see `src/models.py`).
`model=None` defaults to `settings.model_complex`. The RAGAS eval
harness calls `run_pipeline` directly, once per golden-set row, and
forwards `top_k` so the sweep can vary retrieval depth without editing
pipeline source.
