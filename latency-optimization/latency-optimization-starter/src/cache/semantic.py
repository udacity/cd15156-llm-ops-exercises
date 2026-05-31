"""Chroma-backed semantic cache primitives (REQ-070, M15).

Three public functions — ``lookup``, ``store``, ``clear`` — implement the
four-move semantic-cache architecture Module 14 named:

1. Embed the incoming question with the same model used for retrieval.
2. Similarity-search a dedicated ``cache`` collection for the single
   nearest neighbor.
3. If the cosine similarity is at or above the threshold, return the
   cached response (tagged ``cached=True``); otherwise return ``None`` and
   let the caller run the full RAG pipeline.
4. After a miss, ``store`` writes the new question + response back with a
   TTL stamp so future paraphrases can hit it.

The cache lives in the same on-disk Chroma directory as the retrieval
store (``settings.chroma_path``) but in a separate collection
(``COLLECTION_NAME = "cache"``) with ``hnsw:space=cosine`` pinned. The
distance-to-similarity convention is ``1 - distance`` so callers see
similarity on the natural ``[0, 1]`` "higher is better" scale.

The default threshold is :data:`src.constants.CACHE_SIMILARITY_THRESHOLD`
(0.85 at the time of this writing) — the typical retail-FAQ baseline for
sentence-transformer-class embedders. Exercises in M15 sweep this knob
at 0.70 / 0.85 / 0.95 to surface the wrong-answer mode at loose
thresholds.
"""

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from src import constants
from src.config import settings
from src.embedder import embed_query
from src.models import QueryResponse

# Silence onnxruntime's C++ stderr warning on import, matching the
# fd-2 trick in src/store.py. Any importer of src.cache.semantic
# (the M15 walkthrough, M11 RAGAS, M18 gateway) inherits the quiet.
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

COLLECTION_NAME = "cache"


@lru_cache(maxsize=1)
def _client() -> "chromadb.PersistentClient":
    """Private singleton client against ``settings.chroma_path``.

    Mirrors the capstone shape (``project/src/cache/semantic.py``) by
    owning a separate client object rather than reaching into
    ``src.store._client``. Both clients point at the same on-disk
    directory, so the ``scikit_docs`` retrieval collection and the
    ``cache`` collection share one Chroma store.
    """
    Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(settings.chroma_path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _collection() -> Any:
    """Open or create the cache collection with cosine HNSW pinned."""
    return _client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def lookup(
    question: str,
    *,
    threshold: float = constants.CACHE_SIMILARITY_THRESHOLD,
) -> QueryResponse | None:
    """Return a cached response for ``question`` or ``None`` on miss.

    Args:
        question: User's question, raw string (no normalisation).
        threshold: Minimum cosine similarity for a hit. Defaults to
            ``constants.CACHE_SIMILARITY_THRESHOLD`` (0.85). Pass a lower
            value to demonstrate the wrong-answer mode (M15 Exercise 2).

    Returns:
        A ``QueryResponse`` with ``cached=True`` on a hit, ``None`` on a
        miss. Lazy TTL eviction fires inline — if the nearest match is
        older than its ``ttl_s`` metadata, the entry is deleted and the
        function returns ``None`` (miss).
    """
    collection = _collection()
    if collection.count() == 0:
        return None

    embedding = embed_query(question)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=1,
        include=["metadatas", "distances"],
    )
    ids = results["ids"][0]
    if not ids:
        return None

    distance = float(results["distances"][0][0])
    similarity = 1.0 - distance
    if similarity < threshold:
        return None

    metadata = results["metadatas"][0][0]
    ttl_s = int(metadata.get("ttl_s", 0))
    if ttl_s > 0:
        created_at = datetime.fromisoformat(metadata["created_at"])
        age_s = (datetime.now(timezone.utc) - created_at).total_seconds()
        if age_s > ttl_s:
            collection.delete(ids=[ids[0]])
            return None

    payload = json.loads(metadata["response_json"])
    payload["cached"] = True
    return QueryResponse.model_validate(payload)


def store(
    question: str,
    response: QueryResponse,
    *,
    ttl_s: int = 3600,
) -> str:
    """Embed ``question`` and upsert it with ``response`` as cached payload.

    Args:
        question: User's question (used as both the embedding source and
            the ``question`` metadata field).
        response: The ``QueryResponse`` to cache. Serialised to JSON for
            storage; ``cached`` is reset to ``False`` on write so the
            ``True`` only ever appears on the read path.
        ttl_s: Time-to-live in seconds. Default 3,600 (one hour) — the
            retail-FAQ-appropriate starting point Module 14 named. A
            value of ``0`` marks the entry immortal.

    Returns:
        The Chroma id under which the entry was stored.
    """
    embedding = embed_query(question)
    key = f"cache:{uuid.uuid4().hex}"
    payload = response.model_copy(update={"cached": False}).model_dump()
    metadata = {
        "question": question,
        "response_json": json.dumps(payload),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ttl_s": ttl_s,
    }
    _collection().upsert(
        ids=[key],
        embeddings=[embedding],
        documents=[question],
        metadatas=[metadata],
    )
    return key


def clear() -> int:
    """Drop every entry in the cache collection. Returns the count removed."""
    collection = _collection()
    count = collection.count()
    if count == 0:
        return 0
    all_ids = collection.get(include=[])["ids"]
    if all_ids:
        collection.delete(ids=all_ids)
    return count


__all__ = ["lookup", "store", "clear", "COLLECTION_NAME"]
