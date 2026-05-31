"""End-to-end RAG pipeline: retrieve → prompt → generate → respond (REQ-066, M07).

Composes the three single-purpose modules filled by REQ-064 and REQ-065:

- ``embedder.embed_query`` — turn the question into a vector.
- ``store.query`` — top-k cosine search against the ``scikit_docs`` collection.
- ``generator.generate`` — render the system prompt and call OpenAI.

The capstone factored these into ``project/src/rag/{retriever,generator,pipeline}.py``;
the ScikitDocs starter inlines the embed-plus-search half (no separate
``retriever.py``) so each frozen ``src/*.py`` file maps 1:1 to one module's
walkthrough. The composition is otherwise identical — same five RAG stages,
same confidence-from-similarity averaging, same ``QueryResponse`` shape.

Phoenix tracing (REQ-067 / M09) wraps this function from the outside with a
top-level span and child spans for each composed call. Don't import
opentelemetry here — M09 patches the function reference rather than asking
M07 to know about tracing.
"""

from src.config import settings
from src.embedder import embed_query
from src.generator import generate
from src.models import QueryResponse
from src.store import query


def run_pipeline(
    question: str, top_k: int = 5, model: str | None = None
) -> QueryResponse:
    """Run the full RAG pipeline and return a structured ``QueryResponse``.

    Args:
        question: User's question (raw string; sanitization is the gateway's
            job at REQ-071 and REQ-072, not the pipeline's).
        top_k:    Retrieval depth. Defaults to ``constants.DEFAULT_TOP_K``.
        model:    OpenAI model. Defaults to ``settings.model_complex`` /
                  ``constants.MODEL_COMPLEX``.

    Returns:
        ``QueryResponse`` populated with ``answer``, ``sources``,
        ``confidence`` (mean of retrieval similarity scores; ``0.0`` if no
        sources came back), ``model``, ``tokens``, and ``cost_usd``. The
        ``cached`` / ``trace_id`` / ``blocked_by`` fields are left at their
        Pydantic defaults — REQ-070 (cache), REQ-067 (tracing), and REQ-072
        (guardrails) populate them in their respective wrappers.
    """
    chosen_model = model or settings.model_complex
    query_embedding = embed_query(question)
    sources = query(query_embedding, n_results=top_k)
    answer, usage, cost = generate(question, sources, chosen_model)
    confidence = (
        sum(s.similarity_score for s in sources) / len(sources) if sources else 0.0
    )
    return QueryResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        model=chosen_model,
        tokens=usage,
        cost_usd=cost,
    )


__all__ = ["run_pipeline"]
