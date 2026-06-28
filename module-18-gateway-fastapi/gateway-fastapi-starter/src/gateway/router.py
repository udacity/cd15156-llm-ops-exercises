"""Tier dispatch + cache-traced composition for the gateway.

The route handler in :mod:`src.gateway.routes` calls :func:`route_query`,
which is where every capability the starter provides converges:

1. **Look up** the cache (:func:`src.cache.semantic.lookup`).
   A hit returns immediately, before any LLM call is made.
2. **On miss, classify** the question (:func:`src.gateway.classifier.classify`).
3. **Select** the model from the tier (:func:`select_model`, this file).
4. **Run** the traced pipeline (:func:`src.tracing.traced_pipeline`)
   so Phoenix gets one span per stage and the OpenAI auto-instrumentor
   gets to attach token/cost attributes to the ``generate`` span.
5. **Store** the response in the cache so future paraphrases hit.
6. **Log** the cost row (:func:`src.cost.tracker.log_request`)
   — only on miss, because cache hits did not make an LLM call.

The ``client_id`` keyword is the sticky-by-user contract. It threads
from the ``X-Client-Id`` request header through the route handler into
this function; this function does nothing with the value beyond passing
it through, but the contract test in ``tests/test_smoke.py`` pins the
plumbing so a sticky-by-user consumer can rely on it.
"""

from src.cache.semantic import lookup, store
from src.config import settings
from src.cost.tracker import log_request
from src.gateway.classifier import QueryType, classify
from src.models import QueryResponse
from src.tracing import traced_pipeline


def select_model(query_type: QueryType) -> str:
    """Map a classifier label to a concrete OpenAI model name."""
    # TODO(m18-ex1): add the premium-tier dispatch arm (check first — most specific case)
    if query_type == "complex":
        return settings.model_complex
    return settings.model_simple


# TODO(m18-ex3): add ``provider: str = "openai"`` keyword to route_query; when ``provider == "anthropic"`` swap chosen_model to "claude-sonnet-stub" and dispatch through the Anthropic adapter (src.gateway.providers.anthropic.generate) before traced_pipeline
def route_query(
    question: str,
    top_k: int = 5,
    *,
    model: str | None = None,
    client_id: str | None = None,
) -> QueryResponse:
    """Classify → tier-select → cache+trace → log on miss.

    Args:
        question: The user's question.
        top_k: Retrieval depth. Defaults to 5 (``constants.DEFAULT_TOP_K``).
        model: Override the tier-selected model. When ``None`` the
            classifier decides the tier and :func:`select_model` picks
            the model from ``settings``.
        client_id: Optional ``X-Client-Id`` value from the request
            header. Forwarded as request state; a sticky-by-user
            variant assignment reads it for arm selection.

    Returns:
        A :class:`QueryResponse` populated by the cache (on hit) or by
        :func:`traced_pipeline` (on miss). The ``cached`` flag
        distinguishes the two paths so the caller can introspect.
    """
    # Cache lookup runs first so a hit short-circuits before any LLM
    # call. The classifier (one gpt-4o-mini round-trip) only runs on a
    # miss, when we actually need the tier to pick a model and log the
    # cost row. A hit returns the stored response without paying it.
    if settings.enable_semantic_cache:
        hit = lookup(question)
        if hit is not None:
            return hit

    query_type = classify(question)
    chosen_model = model or select_model(query_type)

    response = traced_pipeline(question, top_k=top_k, model=chosen_model)
    if settings.enable_semantic_cache:
        store(question, response)
    log_request(chosen_model, response.tokens, response.cost_usd, query_type)
    return response


__all__ = ["select_model", "route_query"]
