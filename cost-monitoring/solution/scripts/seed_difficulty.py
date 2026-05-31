"""Upsert the deliberately-seeded difficulty chunks into the Chroma collection (REQ-063).

Reads ``data/seeded_chunks.jsonl`` (8 chunks: 3 near-duplicate, 3
version-conflicting, 2 embedding-confusion), embeds them via the same
OpenAI cache as ``scripts/load_data.py``, and upserts them into the
``scikit_docs`` collection with ``is_seeded: true`` in metadata.

Runs after ``make load-data``. Idempotent — re-running upserts the same
eight IDs.

Why these chunks exist: a too-clean documentation corpus produces near-
ceiling recall and makes the M11 RAGAS top-k sweep pedagogically flat.
The 8 seeded chunks (~0.2% of the corpus) reintroduce confusion without
breaking the smoke gate's recall@5 ≥ 0.7 floor. See
``data/SEEDING_NOTES.md`` for the full rationale.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Match load_data.py's fd-2 silencing for onnxruntime's C++-init warning.
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

from scripts.load_data import (  # type: ignore[import-not-found]
    COLLECTION_NAME,
    EMBEDDING_CACHE_PATH,
    _chunk_hash,
    _flatten_metadata,
    _make_openai_client,
    append_to_cache,
    load_embedding_cache,
)
from src import constants  # noqa: F401  (imported for side-effect parity)
from src.config import settings

logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

SEEDED_CHUNKS_PATH: Path = Path("data/seeded_chunks.jsonl")
CHROMA_PATH: Path = Path(settings.chroma_path)


def load_seeded_chunks(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run from module-starters/scikit-docs/")
    chunks: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def embed_seeded(client: OpenAI, chunks: list[dict]) -> list[list[float]]:
    """Embed via the shared cache. 8 chunks fit in one API batch."""
    cache = load_embedding_cache(EMBEDDING_CACHE_PATH)
    hashes = [_chunk_hash(c["text"]) for c in chunks]
    missing_idx = [i for i, h in enumerate(hashes) if h not in cache]
    if missing_idx:
        texts = [chunks[i]["text"] for i in missing_idx]
        response = client.embeddings.create(input=texts, model=settings.embedding_model)
        new_entries: list[tuple[str, list[float]]] = []
        for idx, item in zip(missing_idx, response.data):
            cache[hashes[idx]] = item.embedding
            new_entries.append((hashes[idx], item.embedding))
        append_to_cache(EMBEDDING_CACHE_PATH, new_entries)
    return [cache[h] for h in hashes]


def upsert_seeded(collection: "chromadb.Collection", chunks: list[dict],
                  embeddings: list[list[float]]) -> None:
    collection.upsert(
        ids=[c["doc_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=[_flatten_metadata(c["metadata"]) for c in chunks],
    )


def main() -> None:
    chunks = load_seeded_chunks(SEEDED_CHUNKS_PATH)
    print(f"[seed_difficulty] loaded {len(chunks)} seeded chunks from {SEEDED_CHUNKS_PATH}")

    client = _make_openai_client()
    embeddings = embed_seeded(client, chunks)
    print(f"[seed_difficulty] embedded {len(embeddings)} chunks "
          f"({settings.embedding_model})")

    client_db = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = client_db.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    upsert_seeded(collection, chunks, embeddings)
    seeded_count = collection.get(where={"is_seeded": True}, include=[])["ids"]
    print(f"[seed_difficulty] collection '{COLLECTION_NAME}' now holds "
          f"{len(seeded_count)} seeded chunks "
          f"(total docs: {collection.count()})")


if __name__ == "__main__":
    main()
