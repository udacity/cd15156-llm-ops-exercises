"""Server-Sent Events streaming endpoint for ScikitDocs (REQ-075, M26).

Lives flat under ``src/`` rather than in a separate ``optimization``
subpackage so the starter keeps its "one file per module" convention.
The capstone factored the same functionality into
``project/src/optimization/{streaming,routes}.py``; the starter inlines
both surfaces here.

The route ``POST /query/stream`` is mounted on the gateway app via
:func:`src.gateway.app.create_app` (REQ-071 mounted the blocking
``/query`` + cost dashboard; REQ-075 adds the streaming router as the
third include). The forward-dependency exception is the same one the
gateway docstring already documents: ``src.streaming`` is taught after
``src.gateway`` in the curriculum, but the app is a wiring layer and is
allowed to know about every module.

Two design notes worth naming:

* **Cache bypass.** The streaming endpoint deliberately does not consult
  the M15 cache. A streamed response cannot meaningfully short-circuit
  on a cache hit — the cache stores a fully realized ``QueryResponse``,
  not a token sequence. The blocking ``/query`` route is what learners
  benchmark for cache wins; the streaming route is what they benchmark
  for time-to-first-byte. M26 Common Pitfalls names this as a gotcha so
  no one mistakes a missing speedup for a broken endpoint.

* **Input-guards seam.** :func:`_pre_stream_guards` is a no-op shim at
  REQ-075. M20 (REQ-072) will fill it with the LLM Guard + Presidio
  scanners the capstone uses at
  ``project/src/guardrails/llm_guard/input_guards.py``. The contract:
  the shim returns ``(possibly_redacted_question, blocked_by_reason)``
  — when ``blocked_by_reason`` is ``None`` the question is safe to
  stream; when it starts with ``"prompt_injection"`` the stream is
  short-circuited to a single ``done`` event with ``blocked_by`` set;
  when it starts with ``"pii_redacted:"`` the question has been
  rewritten in place and the stream proceeds with the redacted text.
  Output guards over streamed tokens are intentionally deferred — the
  hallucination and off-topic checks both need the whole answer to fire
  and applying them on a partial token stream is a follow-up exercise.
"""

import json
from typing import Iterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel, Field

from src import constants
from src.config import settings
from src.cost.tracker import log_request
from src.embedder import embed_query
from src.generator import render_system_prompt
from src.models import QueryResponse, Source, TokenUsage
from src.pricing import compute_cost
from src.store import query as store_query

# Note: ``src.gateway.classifier`` and ``src.gateway.router`` are
# imported inside :func:`_stream` rather than at module top level so
# that ``src.streaming`` can be loaded standalone without triggering
# ``src.gateway`` package initialization. The gateway's
# ``__init__.py`` eagerly imports ``src.gateway.app``, which in turn
# imports :data:`streaming_router` — so a top-level import of
# ``src.gateway.classifier`` from this module would close the cycle.
# Deferring to request time is the documented workaround.


_BLOCKED_MESSAGE: str = (
    "This request was blocked by an input guardrail. Please rephrase "
    "your question without prompts that look like instructions to the "
    "assistant."
)


class StreamQueryRequest(BaseModel):
    """Request body for ``POST /query/stream``.

    Same caps as :class:`src.gateway.routes.QueryRequest` — 4,000 chars
    on the question, top_k bounded at 20. Kept as a separate Pydantic
    model so the streaming route can evolve its own validation surface
    (a future ``stream_options`` field, for example) without coupling
    to the blocking route's request shape.
    """

    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(constants.DEFAULT_TOP_K, ge=1, le=20)


def _pre_stream_guards(question: str) -> tuple[str, str | None]:
    """Run input guards before the SSE stream opens. **No-op at M26.**

    REQ-072 (M20 — Guardrails Implementation) fills this shim with the
    LLM Guard prompt-injection scanner + the Presidio PII redactor that
    the capstone uses at
    ``project/src/guardrails/llm_guard/input_guards.py``. Until then
    the shim returns the question unchanged with ``None`` for the
    ``blocked_by`` reason, and the streaming route proceeds straight to
    retrieval + generation.

    The contract M20 will fill against is:

    * ``(question, None)`` — safe; stream the answer.
    * ``(cleaned_question, "pii_redacted: EMAIL,PHONE")`` — PII was
      detected and rewritten in place; stream the answer against the
      cleaned text and surface the redacted entity types on the final
      ``done`` event's ``blocked_by`` field.
    * ``("", "prompt_injection: <reason>")`` — the question is unsafe;
      the route short-circuits to :func:`_blocked_stream` and never
      makes an LLM call.

    Keeping the shim explicit (rather than letting M20 add the call
    site directly) gives M26's demo a concrete file:line to point at
    when walking the input-guards-before-stream design.
    """
    return question, None


def _sse_event(payload: dict) -> str:
    """Serialize ``payload`` as one Server-Sent Event ``data:`` frame."""
    return f"data: {json.dumps(payload)}\n\n"


def _blocked_stream(blocked: QueryResponse) -> Iterator[str]:
    """Yield exactly one ``done`` SSE event for an injection-blocked request.

    Streaming-route equivalent of the blocking route's safe-refusal
    short-circuit: no tokens emitted, no LLM call made, just the safe
    ``QueryResponse`` with ``blocked_by`` populated so the client can
    render the refusal in the same UI surface as a normal answer.
    """
    yield _sse_event(
        {"type": "done", "response": json.loads(blocked.model_dump_json())}
    )


def stream_completion(
    question: str, sources: list[Source], model: str
) -> Iterator[str | tuple[str, TokenUsage, float]]:
    """Yield each token as a ``str``; finally yield ``(answer, usage, cost)``.

    Mirrors ``project/src/optimization/streaming.py:22-62``. The
    ``stream_options={"include_usage": True}`` argument on line 38 of
    the capstone is the load-bearing API choice: without it, the OpenAI
    streaming response omits the ``usage`` field on its final chunk and
    the per-call cost computation has nothing to work from. M26 Demo
    Part 2 names this line explicitly; the Common Pitfalls block names
    forgetting it as the gotcha.

    Args:
        question: User's question (already past the input-guards seam).
        sources:  Retrieved chunks for the system prompt.
        model:    OpenAI model name (gateway tier-selected, typically
                  ``constants.MODEL_COMPLEX`` or ``MODEL_SIMPLE``).

    Yields:
        ``str`` for each incremental token (``delta.content``), then a
        ``(answer, TokenUsage, cost_usd)`` tuple on the final iteration.
    """
    client = OpenAI(base_url=settings.openai_base_url or None)
    system_prompt = render_system_prompt(sources)
    stream = client.chat.completions.create(
        model=model,
        temperature=constants.GENERATION_TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        stream=True,
        stream_options={"include_usage": True},
    )

    parts: list[str] = []
    usage: TokenUsage | None = None

    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta.content
            if delta:
                parts.append(delta)
                yield delta
        if getattr(chunk, "usage", None):
            usage = TokenUsage(
                prompt_tokens=chunk.usage.prompt_tokens,
                completion_tokens=chunk.usage.completion_tokens,
            )

    # Defensive fallback — some self-hosted OpenAI-compatible providers
    # omit usage even when stream_options requests it. Same fallback as
    # capstone streaming.py.
    if usage is None:
        usage = TokenUsage(prompt_tokens=0, completion_tokens=0)

    answer = "".join(parts)
    cost = compute_cost(model, usage)
    yield (answer, usage, cost)


def _stream(question: str, top_k: int, blocked_by: str | None) -> Iterator[str]:
    """SSE generator: tier classify → retrieve → stream tokens → done event."""
    # Lazy import to avoid the import cycle documented at the top of
    # this module: src.gateway.__init__ eagerly imports src.gateway.app
    # which imports streaming_router from here.
    from src.gateway.classifier import classify
    from src.gateway.router import select_model

    query_type = classify(question)
    model = select_model(query_type)

    query_embedding = embed_query(question)
    sources = store_query(query_embedding, n_results=top_k)
    confidence = (
        sum(s.similarity_score for s in sources) / len(sources)
        if sources
        else 0.0
    )

    answer = ""
    usage: TokenUsage | None = None
    cost = 0.0

    for piece in stream_completion(question, sources, model):
        if isinstance(piece, tuple):
            answer, usage, cost = piece
            break
        yield _sse_event({"type": "token", "content": piece})

    if usage is None:
        usage = TokenUsage(prompt_tokens=0, completion_tokens=0)

    response = QueryResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        model=model,
        tokens=usage,
        cost_usd=cost,
        blocked_by=blocked_by,
    )
    # Log on every successful stream — the streaming route bypasses the
    # cache by design, so unlike the blocking router's "log on miss"
    # pattern there's no cache-hit branch to skip the cost row for.
    log_request(model, usage, cost, query_type)
    yield _sse_event(
        {"type": "done", "response": json.loads(response.model_dump_json())}
    )


streaming_router = APIRouter()


@streaming_router.post("/query/stream")
def query_stream(request: StreamQueryRequest) -> StreamingResponse:
    """Stream a ScikitDocs answer token-by-token via Server-Sent Events.

    Pipeline:

    1. :func:`_pre_stream_guards` runs first (no-op at M26 — M20 fills it).
       A prompt-injection match short-circuits to :func:`_blocked_stream`
       and the SSE response carries exactly one ``done`` event; PII in
       the question is redacted in place before retrieval and generation.
    2. The cleaned question goes through the tier classifier + the
       retriever, then ``stream_completion`` opens an OpenAI streaming
       chat completion and yields each token as a typed SSE event.
    3. A final ``done`` event carries the full ``QueryResponse``
       (answer, sources, model, tokens, cost, plus ``blocked_by`` when
       PII was redacted).

    Output guards over streamed tokens are intentionally deferred.
    Applying NLI-based hallucination or banned-topic checks across an
    in-flight stream is a guardrails-composition follow-up; the blocking
    ``/query`` route is where both guards fire on the assembled answer.
    """
    cleaned, blocked_by = _pre_stream_guards(request.question)
    if blocked_by and blocked_by.startswith("prompt_injection"):
        blocked = QueryResponse(
            answer=_BLOCKED_MESSAGE,
            sources=[],
            confidence=0.0,
            model=constants.MODEL_SIMPLE,
            tokens=TokenUsage(prompt_tokens=0, completion_tokens=0),
            cost_usd=0.0,
            blocked_by=blocked_by,
        )
        return StreamingResponse(
            _blocked_stream(blocked),
            media_type="text/event-stream",
        )
    return StreamingResponse(
        _stream(cleaned, request.top_k, blocked_by),
        media_type="text/event-stream",
    )


__all__ = [
    "StreamQueryRequest",
    "stream_completion",
    "streaming_router",
    "query_stream",
]
