"""Chroma vector-store client (Module 05).

Wraps :class:`chromadb.PersistentClient` writing to
``settings.chroma_path``. Default collection name is ``scikit_docs`` (vs.
the capstone's ``products``). The collection is created with
``hnsw:space=cosine`` pinned — Chroma defaults to L2, which silently
corrupts ranking against normalised OpenAI embeddings, so the override
is load-bearing.

``add`` uses :meth:`Collection.upsert` (not ``add``) so re-runs against
the same ids don't raise. ``query`` translates Chroma's cosine
*distance* (``1 - similarity``) into a similarity score on the way out
so callers see the natural "higher = better" direction.
"""

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config import settings
from src.models import Source

# Silence onnxruntime's C++ stderr warning on import. Matches the
# fd-2 trick in scripts/load_data.py — Python logging knobs don't
# reach onnxruntime, and on CPU-only hosts (Workspace included) the
# warning fires every import. Wrapped here so any importer of
# `src.store` (Module 07 pipeline, Module 11 RAGAS) inherits the quiet behaviour.
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


@lru_cache(maxsize=1)
def _client() -> "chromadb.PersistentClient":
    """Lazily construct a singleton Chroma client."""
    Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(settings.chroma_path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


ALIAS_NAME: str = "scikit_docs"
ALIAS_FILE: Path = Path("data/ACTIVE_COLLECTION")


def _resolve_alias(name: str) -> str:
    """If ``name`` is the public alias, resolve to the active color.

    Module 24 introduces a blue/green alias mechanism. The active
    collection name is recorded as one line in ``data/ACTIVE_COLLECTION``
    (e.g., ``scikit_docs_blue``). When the file is missing — the legacy
    pre-Module 24 state, before any migration has run — the alias resolves to
    its own name and ``get_or_create_collection`` returns the original
    ``scikit_docs`` collection so Module 05's behaviour is preserved.
    """
    if name != ALIAS_NAME:
        return name
    if not ALIAS_FILE.exists():
        return ALIAS_NAME
    resolved = ALIAS_FILE.read_text(encoding="utf-8").strip()
    return resolved or ALIAS_NAME


def get_collection(name: str = "scikit_docs") -> Any:
    """Get or create a Chroma collection.

    Args:
        name: Collection name. Defaults to ``"scikit_docs"`` (the public
            alias resolved through :func:`_resolve_alias`). Pass an
            explicit color name (``"scikit_docs_blue"`` /
            ``"scikit_docs_green"``) to bypass the alias — Module 24's migration
            script uses this to build into the inactive color.

    Returns:
        A ``chromadb.Collection`` with ``hnsw:space=cosine`` pinned at
        create time. (Returned as ``Any`` to keep the public signature
        independent of the chromadb type surface.)
    """
    return _client().get_or_create_collection(
        name=_resolve_alias(name),
        metadata={"hnsw:space": "cosine"},
    )


def add(
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
    ids: list[str],
) -> None:
    """Upsert a batch of pre-embedded documents into the default collection.

    Uses :meth:`Collection.upsert` so re-runs against the same ids are
    idempotent (the load-time path re-runs frequently as the corpus
    rebuilds; Module 24's blue/green swap also relies on this).

    Args:
        documents:  Raw chunk text. Same length as ``embeddings``.
        embeddings: One vector per document.
        metadatas:  Per-document scalar-only metadata (lists/dicts must
            already be JSON-serialised by the caller — Chroma rejects
            non-scalar metadata values).
        ids:        Stable per-document ids; used as the dedup key.
    """
    get_collection().upsert(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )


def query(query_embedding: list[float], n_results: int = 5) -> list[Source]:
    """Return the top-N nearest chunks for a query embedding.

    Args:
        query_embedding: A single embedding vector
            (from :func:`src.embedder.embed_query`).
        n_results: Top-K. Defaults to 5 (matches
            ``constants.DEFAULT_TOP_K``).

    Returns:
        list[Source] sorted by ``similarity_score`` descending. The
        similarity score is ``1 - cosine_distance`` so callers get a
        natural "higher = better" value in ``[0, 2]`` (cosine distance
        on un-normalised vectors can exceed 1).
    """
    result = get_collection().query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances"],
    )
    documents = result["documents"][0]
    distances = result["distances"][0]
    ids = result["ids"][0]
    sources = [
        Source(
            doc_id=doc_id,
            chunk_text=document,
            similarity_score=1.0 - distance,
        )
        for doc_id, document, distance in zip(ids, documents, distances)
    ]
    sources.sort(key=lambda s: s.similarity_score, reverse=True)
    return sources
