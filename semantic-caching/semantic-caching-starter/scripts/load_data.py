"""Bootstrap the scikit-learn docs corpus into Chroma.

Usage:
    uv run python scripts/load_data.py
    # or: make load-data

Pipeline:
    1. Ensure ``data/scikit-learn-cache/`` is a shallow clone of
       scikit-learn at ``corpus.SCIKIT_LEARN_TAG``; resolve its git SHA.
    2. Parse every RST under the configured doc subdirs via
       ``src.corpus.load_corpus`` (yields one dict per section).
    3. Section-aware chunking: split sections >512 tokens at paragraph
       boundaries (~450 target, 75-token overlap); drop chunks <50 tokens.
    4. Dedup by ``doc_id`` (last-wins) so re-running is idempotent.
    5. Cache embeddings keyed by SHA-256 of ``chunk_text`` in
       ``data/embedding_cache.jsonl`` — only un-cached chunks hit the API.
    6. Batch the OpenAI embedding API at 256 chunks per request, four
       concurrent in-flight requests, so a cold build of ~4,000 chunks
       fits the <60s Workspace target.
    7. Upsert into the Chroma collection that the ``scikit_docs`` alias
       currently resolves to (Module 24 blue/green) with
       ``hnsw:space=cosine`` pinned at create time. Pre-Module 24 starters and
       fresh checkouts with no ``data/ACTIVE_COLLECTION`` file land in
       the literal ``scikit_docs`` collection — the initial scaffolding's original
       behaviour. Chroma's metadata columns are scalar-only, so
       list-typed fields (xrefs, code_languages) are JSON-serialised on
       the way in.
    8. Write ``data/CORPUS_VERSION`` with tag + SHA + timestamp + chunk
       count so Module 24 RAGOps can read it for blue/green migrations.

Module 05 refactors this script to use the now-filled
``src.chunker.chunk_doc`` / ``src.embedder.embed`` / ``src.store.add``
stubs. Until then the chunk/embed/upsert logic lives inline here.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from src import constants, corpus
from src.config import settings

# Match capstone's chromadb fd-2 silencing — onnxruntime warns at C++
# init on CPU-only hosts (Workspace included) and Python logging knobs
# don't reach it. See project/src/vectordb/store.py.
_saved_fd2 = os.dup(2)
_devnull = os.open(os.devnull, os.O_WRONLY)
try:
    os.dup2(_devnull, 2)
    import chromadb
    from chromadb.config import Settings as ChromaSettings
finally:
    sys.stderr.flush()
    os.dup2(_saved_fd2, 2)
    os.close(_devnull)
    os.close(_saved_fd2)

import logging

from openai import OpenAI

logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


# === Constants tuned for the <60s cold-build target ============================

CHUNK_TARGET_TOKENS: int = 450  # within constants.CHUNK_TARGET_TOKENS guidance
CHUNK_MAX_TOKENS: int = 512
CHUNK_MIN_TOKENS: int = 50
CHUNK_OVERLAP_TOKENS: int = 75

EMBED_BATCH_SIZE: int = 256
EMBED_PARALLELISM: int = 4
UPSERT_BATCH_SIZE: int = 500

COLLECTION_NAME: str = "scikit_docs"

REPO_CACHE_DIR: Path = Path("data/scikit-learn-cache")
EMBEDDING_CACHE_PATH: Path = Path("data/embedding_cache.jsonl")
CORPUS_VERSION_PATH: Path = Path("data/CORPUS_VERSION")
CHROMA_PATH: Path = Path(settings.chroma_path)


# === Repo cache management =====================================================


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def ensure_repo_cache(repo_dir: Path, tag: str) -> str:
    """Idempotently shallow-clone scikit-learn at ``tag``; return resolved SHA."""
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if not (repo_dir / ".git").exists():
        print(f"[load_data] cloning scikit-learn@{tag} → {repo_dir} ...")
        _run([
            "git", "clone", "--depth=1", "--branch", tag,
            corpus.SCIKIT_LEARN_REPO_URL, str(repo_dir),
        ])
    else:
        # Already cloned. Verify the tag we have matches.
        current = _run(["git", "rev-parse", "HEAD"], cwd=repo_dir)
        try:
            wanted = _run(["git", "rev-parse", f"refs/tags/{tag}"], cwd=repo_dir)
        except subprocess.CalledProcessError:
            wanted = current  # tag unknown locally; trust HEAD
        if current != wanted:
            print(f"[load_data] re-checking out scikit-learn@{tag} ...")
            _run([
                "git", "fetch", "--depth=1", "origin",
                f"refs/tags/{tag}:refs/tags/{tag}",
            ], cwd=repo_dir)
            _run(["git", "checkout", f"refs/tags/{tag}"], cwd=repo_dir)
    return _run(["git", "rev-parse", "HEAD"], cwd=repo_dir)


# === Chunking ==================================================================


def _split_long_text(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Split ``text`` at paragraph boundaries, target ``CHUNK_TARGET_TOKENS``."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for paragraph in paragraphs:
        ptokens = corpus.token_count(paragraph)
        if ptokens > max_tokens:
            # A single paragraph that's already too large — emit what we have
            # then take the paragraph as-is. Splitting mid-paragraph would
            # hurt embedding semantics more than going slightly over budget.
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_tokens = 0
            chunks.append(paragraph)
            continue
        if current_tokens + ptokens > max_tokens:
            chunks.append("\n\n".join(current))
            tail = current[-1] if current else ""
            tail_tokens = corpus.token_count(tail)
            if tail_tokens <= overlap:
                current = [tail, paragraph]
                current_tokens = tail_tokens + ptokens
            else:
                current = [paragraph]
                current_tokens = ptokens
        else:
            current.append(paragraph)
            current_tokens += ptokens
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_section(section: dict) -> list[dict]:
    """Yield chunk dicts for ``section``, splitting if it exceeds the budget."""
    text = section["text"]
    tokens = corpus.token_count(text)
    if tokens < CHUNK_MIN_TOKENS:
        return []
    if tokens <= CHUNK_MAX_TOKENS:
        return [{
            "doc_id": section["doc_id"],
            "text": text,
            "metadata": dict(section["metadata"]),
        }]
    pieces = _split_long_text(text, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS)
    chunks = []
    for idx, piece in enumerate(pieces):
        if corpus.token_count(piece) < CHUNK_MIN_TOKENS:
            continue
        chunks.append({
            "doc_id": f"{section['doc_id']}.p{idx}",
            "text": piece,
            "metadata": dict(section["metadata"]),
        })
    return chunks


# === Embedding cache ===========================================================


def _chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_embedding_cache(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    cache: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            cache[entry["hash"]] = entry["embedding"]
    return cache


def append_to_cache(path: Path, entries: list[tuple[str, list[float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for h, emb in entries:
            fh.write(json.dumps({"hash": h, "embedding": emb}) + "\n")


# === Embedding ================================================================


def _make_openai_client() -> OpenAI:
    api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
    base_url = settings.openai_base_url or os.environ.get(constants.OPENAI_BASE_URL_ENV)
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY missing. Copy .env.example to .env and set the key."
        )
    return OpenAI(api_key=api_key, base_url=base_url or None)


def _embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        input=texts, model=settings.embedding_model,
    )
    return [item.embedding for item in response.data]


def embed_missing(
    client: OpenAI,
    chunks: list[dict],
    cache: dict[str, list[float]],
) -> list[list[float]]:
    """Return embeddings for ``chunks``, hitting the cache where possible."""
    hashes = [_chunk_hash(c["text"]) for c in chunks]
    missing_idx = [i for i, h in enumerate(hashes) if h not in cache]
    if missing_idx:
        batches = [
            missing_idx[i : i + EMBED_BATCH_SIZE]
            for i in range(0, len(missing_idx), EMBED_BATCH_SIZE)
        ]
        with ThreadPoolExecutor(max_workers=EMBED_PARALLELISM) as pool:
            futures = {
                pool.submit(_embed_batch, client, [chunks[j]["text"] for j in batch]): batch
                for batch in batches
            }
            new_entries: list[tuple[str, list[float]]] = []
            for fut in futures:
                batch = futures[fut]
                vectors = fut.result()
                for j, vec in zip(batch, vectors):
                    cache[hashes[j]] = vec
                    new_entries.append((hashes[j], vec))
        append_to_cache(EMBEDDING_CACHE_PATH, new_entries)
    return [cache[h] for h in hashes]


# === Chroma upsert =============================================================


def _make_chroma_collection() -> "chromadb.Collection":
    """Open the active collection by routing through :func:`src.store.get_collection`.

    Module 24 introduced the blue/green alias mechanism — passing
    ``COLLECTION_NAME`` (``"scikit_docs"``, the public alias) through
    ``store.get_collection`` means a post-migration ``make load-data``
    refreshes whichever color the alias currently names rather than
    silently writing to the pre-alias ``scikit_docs`` collection. When
    no ``data/ACTIVE_COLLECTION`` file exists (bootstrap / pre-Module 24
    starter) the resolver returns the literal ``scikit_docs`` and
    behaviour matches the initial scaffolding's original code path.
    """
    from src import store

    return store.get_collection(COLLECTION_NAME)


def _flatten_metadata(metadata: dict) -> dict:
    """Chroma metadata is scalar-only; serialize list/dict fields to JSON."""
    flat: dict = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            flat[key] = value
        else:
            flat[key] = json.dumps(value)
    return flat


def upsert_chunks(
    collection: "chromadb.Collection",
    chunks: list[dict],
    embeddings: list[list[float]],
) -> None:
    for i in range(0, len(chunks), UPSERT_BATCH_SIZE):
        batch = chunks[i : i + UPSERT_BATCH_SIZE]
        batch_embeddings = embeddings[i : i + UPSERT_BATCH_SIZE]
        collection.upsert(
            ids=[c["doc_id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=batch_embeddings,
            metadatas=[_flatten_metadata(c["metadata"]) for c in batch],
        )


# === Top-level orchestration ===================================================


def write_corpus_version(
    path: Path, tag: str, sha: str, chunk_count: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).isoformat(timespec="seconds")
    path.write_text(
        f"scikit_learn_tag={tag}\n"
        f"scikit_learn_sha={sha}\n"
        f"ingest_timestamp={timestamp}\n"
        f"chunk_count={chunk_count}\n",
        encoding="utf-8",
    )


def main() -> None:
    start = time.monotonic()
    tag = corpus.SCIKIT_LEARN_TAG
    sha = ensure_repo_cache(REPO_CACHE_DIR, tag)
    print(f"[load_data] scikit-learn@{tag} resolved to {sha[:12]}")

    print(f"[load_data] parsing RST sources under {REPO_CACHE_DIR}/doc/ ...")
    parse_start = time.monotonic()
    seen_doc_ids: set[str] = set()
    chunks: list[dict] = []
    for section in corpus.load_corpus(REPO_CACHE_DIR, sha):
        for chunk in chunk_section(section):
            doc_id = chunk["doc_id"]
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            chunks.append(chunk)
    print(
        f"[load_data] parsed {len(chunks)} chunks "
        f"in {time.monotonic() - parse_start:.1f}s"
    )

    if not chunks:
        raise SystemExit("No chunks produced — corpus is empty.")

    print(f"[load_data] embedding {len(chunks)} chunks "
          f"(batch={EMBED_BATCH_SIZE}, parallel={EMBED_PARALLELISM}) ...")
    embed_start = time.monotonic()
    cache = load_embedding_cache(EMBEDDING_CACHE_PATH)
    cache_hits_before = len(cache)
    client = _make_openai_client()
    embeddings = embed_missing(client, chunks, cache)
    new_embeds = len(cache) - cache_hits_before
    print(
        f"[load_data] embedded {new_embeds} new + reused "
        f"{len(chunks) - new_embeds} cached in "
        f"{time.monotonic() - embed_start:.1f}s"
    )

    collection = _make_chroma_collection()
    print(f"[load_data] upserting into Chroma collection '{collection.name}' ...")
    upsert_chunks(collection, chunks, embeddings)

    write_corpus_version(CORPUS_VERSION_PATH, tag, sha, len(chunks))
    print(
        f"[load_data] done — {len(chunks)} chunks upserted, "
        f"total {time.monotonic() - start:.1f}s"
    )


if __name__ == "__main__":
    main()
