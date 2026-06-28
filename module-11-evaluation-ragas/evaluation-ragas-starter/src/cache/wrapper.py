"""Cache-then-route composition for the Module 15 demo + exercises.

The capstone's HTTP route at ``project/src/gateway/routes.py`` reproduces
this same shape inline, between the input guards and the output guards.
Until Module 18 ships the gateway in the starter, Module 15 calls
:func:`cached_route_query` directly from Python so learners can run the
demo and exercises without standing up a FastAPI server.

The composition has three steps:

1. ``lookup`` the question against the cache; if it hits at the given
   threshold, return immediately (with ``cached=True``).
2. On a miss, call :func:`src.pipeline.run_pipeline` to produce a fresh
   answer.
3. ``store`` the new response back under the original question so future
   paraphrases at or above the threshold hit it.
"""

from src import constants
from src.cache.semantic import lookup, store
from src.config import settings
from src.models import QueryResponse
from src.pipeline import run_pipeline


def cached_route_query(
    question: str,
    top_k: int = constants.DEFAULT_TOP_K,
    *,
    threshold: float = constants.CACHE_SIMILARITY_THRESHOLD,
    ttl_s: int = 3600,
    model: str | None = None,
) -> QueryResponse:
    """Look the question up in the cache; on miss, run the pipeline and store.

    Args:
        question: User's question.
        top_k: Retrieval depth on miss. Defaults to
            ``constants.DEFAULT_TOP_K`` (5).
        threshold: Cosine-similarity threshold for the cache. Defaults to
            ``constants.CACHE_SIMILARITY_THRESHOLD`` (0.85).
        ttl_s: TTL for newly stored entries. Defaults to 3,600 seconds.
        model: Optional OpenAI model override; passed through to
            :func:`run_pipeline`.

    Returns:
        A ``QueryResponse``. ``response.cached`` is ``True`` on a hit and
        ``False`` on a miss (the freshly generated response).
    """
    if settings.enable_semantic_cache:
        hit = lookup(question, threshold=threshold)
        if hit is not None:
            return hit

    response = run_pipeline(question, top_k=top_k, model=model)
    if settings.enable_semantic_cache:
        store(question, response, ttl_s=ttl_s)
    return response


__all__ = ["cached_route_query"]
