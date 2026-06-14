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
    # Premium tier first — most specific case, falls through to complex then simple.
    if query_type == "premium":
        return settings.model_premium
    if query_type == "complex":
        return settings.model_complex
    return settings.model_simple


# ``provider`` keyword selects the adapter — "openai" uses traced_pipeline, "anthropic" swaps to the stub.
def route_query(
    question: str,
    top_k: int = 5,
    *,
    model: str | None = None,
    client_id: str | None = None,
    provider: str = "openai",
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
        provider: ``"openai"`` (default) or ``"anthropic"`` — Exercise 3
            adds the Anthropic adapter dispatch arm.

    Returns:
        A :class:`QueryResponse` populated by the cache (on hit) or by
        :func:`traced_pipeline` (on miss). The ``cached`` flag
        distinguishes the two paths so the caller can introspect.
    """
    # Cache lookup runs first so a hit short-circuits before any LLM
    # call. The classifier (one gpt-4o-mini round-trip) only runs on a
    # miss, when we actually need the tier to pick a model and log the
    # cost row. A hit returns the stored response without paying it.
    hit = lookup(question)
    if hit is not None:
        return hit

    query_type = classify(question)
    chosen_model = model or select_model(query_type)
    if provider == "anthropic":
        # Exercise 3 — synthetic model name keyed in src/pricing.py so
        # compute_cost has a rate to use. A real implementation would
        # route through Settings.anthropic_model_complex / _simple per
        # tier, mirroring the OpenAI side.
        chosen_model = "claude-sonnet-stub"

    if provider == "anthropic":
        # Local imports keep the OpenAI-only path import-clean — the
        # Anthropic adapter is opt-in.
        from src.embedder import embed_query as embed
        from src.gateway.providers.anthropic import generate as anthropic_generate
        from src.generator import render_system_prompt
        from src.store import query as store_query

        sources = store_query(embed(question), n_results=top_k)
        system_prompt = render_system_prompt(sources)
        answer, usage, cost = anthropic_generate(question, system_prompt, chosen_model)
        response = QueryResponse(
            answer=answer,
            sources=sources,
            confidence=sum(s.similarity_score for s in sources) / max(len(sources), 1),
            model=chosen_model,
            tokens=usage,
            cost_usd=cost,
        )
    else:
        response = traced_pipeline(question, top_k=top_k, model=chosen_model)

    store(question, response)
    log_request(chosen_model, response.tokens, response.cost_usd, query_type)
    return response


__all__ = ["select_model", "route_query"]
