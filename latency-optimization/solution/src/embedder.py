"""OpenAI embeddings wrapper (REQ-065, M05).

Single-string and list-of-strings inputs share one entry point. List
inputs are sent in batches of ``_BATCH_SIZE`` per OpenAI request — the
key performance lever for cold loads (a 4,000-chunk corpus takes ~30s
batched at 256 vs ~13 minutes one-at-a-time). The batch size matches
:mod:`scripts.load_data`'s ``EMBED_BATCH_SIZE``.

Reads ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` from settings/env so the
same code path works against api.openai.com and the Vocareum proxy.
"""

import os
from functools import lru_cache

from openai import OpenAI

from src import constants
from src.config import settings

# Single batched OpenAI request per N chunks. Matches the load-time
# target documented in INTERFACES.md ("≥256 per request"). The OpenAI
# embeddings endpoint accepts up to 2048 inputs per call, but 256 is
# the elbow where added bytes per request stop reducing round-trip cost.
_BATCH_SIZE: int = 256


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    """Lazily build a singleton OpenAI client."""
    api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
    base_url = settings.openai_base_url or os.environ.get(constants.OPENAI_BASE_URL_ENV)
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY missing. Copy .env.example to .env and set the key."
        )
    return OpenAI(api_key=api_key, base_url=base_url or None)


def _embed_batch(texts: list[str]) -> list[list[float]]:
    response = _client().embeddings.create(
        input=texts, model=settings.embedding_model,
    )
    return [item.embedding for item in response.data]


def embed(text: str | list[str]) -> list[float] | list[list[float]]:
    """Embed text via OpenAI; returns one vector per input.

    Args:
        text: A single string or a list of strings. Lists are sent in
            batches of ``_BATCH_SIZE``; a single OpenAI request per batch.

    Returns:
        list[float] when ``text`` is a string;
        list[list[float]] when ``text`` is a list (one vector per input,
        in input order).
    """
    if isinstance(text, str):
        return _embed_batch([text])[0]
    if not text:
        return []
    vectors: list[list[float]] = []
    for start in range(0, len(text), _BATCH_SIZE):
        vectors.extend(_embed_batch(text[start : start + _BATCH_SIZE]))
    return vectors


def embed_query(text: str) -> list[float]:
    """Convenience wrapper for a single query embedding.

    Args:
        text: The query string.

    Returns:
        list[float] of length ``constants.EMBEDDING_DIM``.
    """
    return _embed_batch([text])[0]
