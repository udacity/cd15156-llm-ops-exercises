"""Tier dispatch + cache-traced composition for the gateway.

The route handler in :mod:`src.gateway.routes` calls :func:`route_query`,
which is where every Wave 1-3 capability the starter shipped converges:

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

The ``client_id`` keyword is the sticky-by-user
contract. It threads from the ``X-Client-Id`` request header through
the route handler into this function; the router itself does nothing
with the value beyond passing it through, but the contract test in
``tests/test_smoke.py`` pins the plumbing so the A/B layer can rely on it.
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
    if query_type == "premium":
        return settings.model_premium
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
            header. The gateway forwards it as request state; the A/B
            layer reads it for sticky-by-user variant assignment.
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
    if settings.enable_semantic_cache:
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
        # TODO(m20-exercise-3): forward MAX_OUTPUT_TOKENS from src.guardrails.rate_limit into the traced_pipeline call so every gateway request carries the output cap
        response = traced_pipeline(question, top_k=top_k, model=chosen_model)

    if settings.enable_semantic_cache:
        store(question, response)
    log_request(chosen_model, response.tokens, response.cost_usd, query_type)
    return response


__all__ = ["select_model", "route_query"]
