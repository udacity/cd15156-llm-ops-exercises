"""Unit tests for the semantic cache primitives (REQ-070, M15).

Mirrors the shape of ``project/tests/cache/test_semantic.py`` but stubs
the embedder and the Chroma client so the suite is hermetic — no OpenAI
calls, no Chroma writes to the real ``data/chroma`` directory. The
fixture-injected fake client mimics the subset of the Chroma collection
API that ``src.cache.semantic`` reaches for.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from src.cache import semantic as cache_module
from src.models import QueryResponse, Source, TokenUsage


# --- Fake Chroma collection --------------------------------------------------


class _FakeCollection:
    """In-memory stand-in for ``chromadb.Collection``.

    Records the metadata + embedding for each upsert and returns the
    nearest neighbour by cosine distance on ``query``. ``n_results=1``
    is enough for the cache's gating logic, so we don't bother sorting.
    """

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}

    def count(self) -> int:
        return len(self._rows)

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        for i, e, d, m in zip(ids, embeddings, documents, metadatas):
            self._rows[i] = {"embedding": e, "document": d, "metadata": m}

    def get(self, include: list[str] | None = None) -> dict[str, list]:
        return {"ids": list(self._rows.keys())}

    def delete(self, ids: list[str]) -> None:
        for i in ids:
            self._rows.pop(i, None)

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str] | None = None,
    ) -> dict:
        if not self._rows:
            return {"ids": [[]], "distances": [[]], "metadatas": [[]]}
        target = query_embeddings[0]

        def _cosine_distance(v: list[float]) -> float:
            # Vectors in this test suite are pre-normalised, so cosine
            # distance is just 1 - dot product.
            dot = sum(a * b for a, b in zip(target, v))
            return 1.0 - dot

        scored = sorted(
            self._rows.items(),
            key=lambda kv: _cosine_distance(kv[1]["embedding"]),
        )[:n_results]
        ids = [k for k, _ in scored]
        distances = [_cosine_distance(v["embedding"]) for _, v in scored]
        metadatas = [v["metadata"] for _, v in scored]
        return {
            "ids": [ids],
            "distances": [distances],
            "metadatas": [metadatas],
        }


@pytest.fixture
def fake_collection(monkeypatch: pytest.MonkeyPatch) -> _FakeCollection:
    """Replace the Chroma collection with an in-memory fake."""
    coll = _FakeCollection()
    monkeypatch.setattr(cache_module, "_collection", lambda: coll)
    return coll


# --- Fake embedder -----------------------------------------------------------

# A deterministic mapping from question → unit vector. Picking the
# coordinates by hand lets the tests assert specific similarity scores.
_EMBEDDINGS: dict[str, list[float]] = {
    # Two queries with identical embeddings → similarity 1.0.
    "warmup": [1.0, 0.0, 0.0],
    "paraphrase_exact": [1.0, 0.0, 0.0],
    # Close but not identical → similarity 0.9.
    "paraphrase_close": [0.9, 0.4358898943540674, 0.0],
    # Orthogonal → similarity 0.0.
    "different_topic": [0.0, 1.0, 0.0],
}


@pytest.fixture
def fake_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``embed_query`` with a lookup against ``_EMBEDDINGS``."""

    def _embed(text: str) -> list[float]:
        return list(_EMBEDDINGS[text])

    monkeypatch.setattr(cache_module, "embed_query", _embed)


# --- QueryResponse helper ----------------------------------------------------


def _make_response(answer: str = "Cached answer.") -> QueryResponse:
    return QueryResponse(
        answer=answer,
        citations=[Source(doc_id="d1", chunk_text="...", similarity_score=0.95)],
        confidence=0.95,
        model="gpt-4o",
        tokens=TokenUsage(prompt_tokens=100, completion_tokens=20),
        cost_usd=0.0008,
    )


# --- Tests -------------------------------------------------------------------


def test_lookup_returns_none_on_empty_cache(
    fake_collection: _FakeCollection, fake_embedder: None
) -> None:
    """Cold-start short-circuit — no embed call, no query, returns None."""
    assert cache_module.lookup("warmup") is None


def test_store_then_lookup_returns_cached_response(
    fake_collection: _FakeCollection, fake_embedder: None
) -> None:
    """Round-trip: store a response, lookup the same question, get it back tagged cached."""
    original = _make_response("squared_error is the default criterion.")
    cache_module.store("warmup", original)

    hit = cache_module.lookup("warmup")
    assert hit is not None
    assert hit.answer == "squared_error is the default criterion."
    assert hit.cached is True


def test_paraphrase_above_threshold_hits(
    fake_collection: _FakeCollection, fake_embedder: None
) -> None:
    """A paraphrase at 0.9 similarity hits at the 0.85 default threshold."""
    cache_module.store("warmup", _make_response())
    hit = cache_module.lookup("paraphrase_close")
    assert hit is not None
    assert hit.cached is True


def test_high_threshold_misses_on_paraphrase(
    fake_collection: _FakeCollection, fake_embedder: None
) -> None:
    """The same paraphrase at 0.9 similarity misses at threshold=0.95."""
    cache_module.store("warmup", _make_response())
    assert cache_module.lookup("paraphrase_close", threshold=0.95) is None


def test_loose_threshold_lets_unrelated_query_through(
    fake_collection: _FakeCollection, fake_embedder: None
) -> None:
    """Orthogonal query at threshold=0.0 hits — the wrong-answer mode at loose thresholds."""
    cache_module.store("warmup", _make_response("Bantam content."))
    hit = cache_module.lookup("different_topic", threshold=0.0)
    assert hit is not None
    assert hit.answer == "Bantam content."  # served the wrong answer; the M15 lift


def test_default_threshold_misses_on_unrelated_query(
    fake_collection: _FakeCollection, fake_embedder: None
) -> None:
    """Orthogonal query correctly misses at the default 0.85 threshold."""
    cache_module.store("warmup", _make_response())
    assert cache_module.lookup("different_topic") is None


def test_clear_removes_all_entries(
    fake_collection: _FakeCollection, fake_embedder: None
) -> None:
    """``clear`` deletes every entry and reports the count removed."""
    cache_module.store("warmup", _make_response())
    cache_module.store("paraphrase_close", _make_response())
    assert fake_collection.count() == 2

    removed = cache_module.clear()
    assert removed == 2
    assert fake_collection.count() == 0
    assert cache_module.clear() == 0  # idempotent


def test_ttl_zero_means_immortal(
    fake_collection: _FakeCollection, fake_embedder: None
) -> None:
    """``ttl_s=0`` entries are never evicted by the lazy TTL path."""
    cache_module.store("warmup", _make_response(), ttl_s=0)
    # Even an entry stamped "two days ago" survives when ttl_s is zero.
    key = next(iter(fake_collection._rows))
    fake_collection._rows[key]["metadata"]["created_at"] = (
        datetime.now(timezone.utc) - timedelta(days=2)
    ).isoformat()
    assert cache_module.lookup("warmup") is not None


def test_ttl_evicts_stale_entry_and_returns_miss(
    fake_collection: _FakeCollection, fake_embedder: None
) -> None:
    """Lazy TTL: an entry older than its ``ttl_s`` is deleted inline + misses."""
    cache_module.store("warmup", _make_response(), ttl_s=60)
    key = next(iter(fake_collection._rows))
    fake_collection._rows[key]["metadata"]["created_at"] = (
        datetime.now(timezone.utc) - timedelta(seconds=120)
    ).isoformat()
    assert cache_module.lookup("warmup") is None
    assert fake_collection.count() == 0  # evicted inline


def test_fresh_entry_within_ttl_returns_hit(
    fake_collection: _FakeCollection, fake_embedder: None
) -> None:
    """An entry younger than its ``ttl_s`` returns a hit and stays in the collection."""
    cache_module.store("warmup", _make_response(), ttl_s=600)
    assert cache_module.lookup("warmup") is not None
    assert fake_collection.count() == 1  # not evicted


def test_store_returns_cache_prefixed_key(
    fake_collection: _FakeCollection, fake_embedder: None
) -> None:
    """The returned id is a ``cache:<uuid>`` string the rubric §10 evidence pattern matches."""
    key = cache_module.store("warmup", _make_response())
    assert key.startswith("cache:")
    assert key in fake_collection._rows


def test_store_strips_cached_true_before_writing(
    fake_collection: _FakeCollection, fake_embedder: None
) -> None:
    """A response with ``cached=True`` is rewritten as ``cached=False`` in the stored JSON.

    Guards against the lookup→store→lookup round-trip flipping the flag
    permanently. ``cached`` should only ever be ``True`` on the read
    path, where ``lookup`` sets it explicitly.
    """
    response_already_cached = _make_response().model_copy(update={"cached": True})
    cache_module.store("warmup", response_already_cached)

    key = next(iter(fake_collection._rows))
    stored_payload = json.loads(
        fake_collection._rows[key]["metadata"]["response_json"]
    )
    assert stored_payload["cached"] is False
