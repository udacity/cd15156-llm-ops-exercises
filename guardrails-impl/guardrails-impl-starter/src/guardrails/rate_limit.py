"""LLM10 Unbounded Consumption — slot 4.

Two mechanisms, both anchored on CVE-2025-53773 (the August 2025 GitHub
Copilot RCE / cost-amplification incident, where prompt injection in
indexed repos coerced the assistant into running long code-generation
loops):

1. :data:`MAX_OUTPUT_TOKENS` — a per-request cap the gateway passes to
   ``generate(...)`` so any single answer is bounded. The default 1024
   covers paragraph-length scikit-learn answers; production teams tune
   this per-route and per-tier.
2. :func:`check_rate_limit` — a token-bucket request limiter keyed by
   ``client_id`` (the :data:`constants.CLIENT_ID_HEADER` value).
   Default budget: 20 requests per 60-second window. When the burst
   exceeds the bucket, the next request blocks with reason string
   ``"unbounded_consumption: rate limit exceeded"``.

The bucket lives in module-level state on purpose — a starter
deployment runs a single uvicorn worker on the workspace, so process-
local state is enough to teach the concept. Production teams swap this
for Redis or a managed limiter; the docstring on each function names
the swap point.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque

from src import constants


MAX_OUTPUT_TOKENS: int = 1024
"""Per-request output-token cap. Passed to ``generate(..., model=..., max_tokens=)``
so a long-generation attack cannot drive a single response past the
bound. Tunable per-route in production — chat routes typically run
higher, doc-Q&A routes (this one) lower.
"""


RATE_LIMIT_REQUESTS: int = 20
"""Bucket capacity. The 21st request inside a single window blocks."""


RATE_LIMIT_WINDOW_SECONDS: float = 60.0
"""Rolling window length. A request is "in the window" if its timestamp
is within ``RATE_LIMIT_WINDOW_SECONDS`` of ``time.monotonic()`` now.
"""


_BUCKETS: dict[str, Deque[float]] = {}
"""Per-client request timestamps. ``client_id="anonymous"`` is the bucket
for un-headered requests — un-headered traffic still counts against a
shared bucket so an unauthenticated burst can not bypass the limit.
"""


_LOCK = threading.Lock()


def _client_bucket(client_id: str | None) -> Deque[float]:
    """Resolve the bucket for ``client_id``, creating it on first call."""
    key = client_id or "anonymous"
    bucket = _BUCKETS.get(key)
    if bucket is None:
        bucket = deque()
        _BUCKETS[key] = bucket
    return bucket


def check_rate_limit(client_id: str | None, *, now: float | None = None) -> str | None:
    """Allow or block a request from ``client_id``.

    Args:
        client_id: The :data:`constants.CLIENT_ID_HEADER` value extracted
            from the request, or ``None`` for un-headered traffic
            (bucketed under ``"anonymous"``).
        now: Override for the current monotonic timestamp. Tests inject a
            controlled clock; production calls pass ``None`` and the
            function reads ``time.monotonic()``.

    Returns:
        ``None`` if the request fits in the bucket (and the timestamp
        has been recorded). A reason string starting with
        ``"unbounded_consumption: "`` if the burst exceeded the bucket
        — the caller returns :func:`src.guardrails.wrapper.safe_response`
        with that reason.

    The function mutates the bucket on the allow path (records the
    timestamp) and does not mutate on the block path (so a sustained
    attacker burst does not extend the window indefinitely). That ordering
    matches the standard token-bucket semantics.
    """
    timestamp = time.monotonic() if now is None else now
    with _LOCK:
        bucket = _client_bucket(client_id)
        cutoff = timestamp - RATE_LIMIT_WINDOW_SECONDS
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_REQUESTS:
            return (
                f"unbounded_consumption: rate limit exceeded "
                f"({RATE_LIMIT_REQUESTS} requests per {int(RATE_LIMIT_WINDOW_SECONDS)}s)"
            )
        bucket.append(timestamp)
    return None


def reset_rate_limit_state() -> None:
    """Clear every bucket. Tests call this to start from a known state."""
    with _LOCK:
        _BUCKETS.clear()


__all__ = [
    "MAX_OUTPUT_TOKENS",
    "RATE_LIMIT_REQUESTS",
    "RATE_LIMIT_WINDOW_SECONDS",
    "check_rate_limit",
    "reset_rate_limit_state",
]


# Constant accessed by reference (so tests can confirm header name doesn't drift).
_HEADER_REF = constants.CLIENT_ID_HEADER
