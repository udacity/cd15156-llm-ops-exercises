"""HTTP route handlers for the ScikitDocs gateway (Module 18 + Module 20).

``POST /query`` accepts a Pydantic-validated body plus the optional
``X-Client-Id`` request header (per ``constants.CLIENT_ID_HEADER``).
``GET /health`` is a static liveness probe so the runtime check has a
zero-cost endpoint to ping.

Module 20 inserted the guardrail stack between the route handler
and :func:`src.gateway.router.route_query`. The order is deliberate
(documented in the handler body): rate-limit first (LLM10 — cheapest
check, fails before any work), prompt-injection regex/DeBERTa second,
system-prompt-leak regex third, PII redaction fourth (passes redacted
text downstream rather than blocking), then the dispatch. The output
guard (LLM-judge hallucination check) runs after dispatch.

Why ``X-Client-Id`` is optional: the gateway must accept the header
when the initial scaffolding sends it for sticky-by-user bucketing, and it must also
serve clean traffic from callers that do not provide one (the Module 11
RAGAS eval harness, for example, fires un-headered requests). Pydantic
+ FastAPI's ``Header(default=None)`` gives both behaviors in one line.
"""

from typing import Annotated

from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from src import constants
from src.gateway.router import route_query
from src.guardrails.input_guards import (
    detect_pii,
    detect_prompt_injection,
    detect_system_prompt_leak,
)
from src.guardrails.llm_judge.output_guards import check_hallucination
from src.guardrails.rate_limit import check_rate_limit
from src.guardrails.wrapper import (
    SAFE_BLOCKED_MESSAGE,
    SAFE_FILTERED_MESSAGE,
    safe_response,
)
from src.models import QueryResponse


class QueryRequest(BaseModel):
    """Validated request body for ``POST /query``.

    The ``question`` cap mirrors the capstone's 4,000-char ceiling —
    enough room for paragraph-length queries, well under the
    context-window budget for either tier. ``top_k`` is bounded to keep
    Chroma latency predictable; widening it past 20 in practice trades
    off retrieval signal-to-noise for chunk-budget pressure on the
    system prompt.
    """

    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(constants.DEFAULT_TOP_K, ge=1, le=20)


router = APIRouter()


@router.post(constants.QUERY_ROUTE, response_model=QueryResponse)
def query_endpoint(
    request: QueryRequest,
    client_id: Annotated[
        str | None, Header(alias=constants.CLIENT_ID_HEADER)
    ] = None,
) -> QueryResponse | JSONResponse:
    """Dispatch through the guardrail stack to :func:`route_query`.

    Order is intentional and load-bearing:

    1. **LLM10 rate limit** — cheapest possible check (one dict lookup +
       a deque operation). Drops abusive bursts before any real work.
    2. **Prompt injection** (regex) — short-circuits the cheap attacks
       so DeBERTa is not invoked on every ``"ignore previous"`` payload.
    3. **System-prompt leak** (regex) — sibling of prompt injection but
       a distinct slot; runs after injection so a hijack-then-extract
       payload is caught by whichever pattern fires first.
    4. **PII redaction** — does not block, returns redacted text. The
       cleaned text flows into ``route_query``; the raw PII never
       reaches the retriever, the cache, or the LLM.
    5. **Dispatch** — :func:`route_query` handles classify → cache →
       traced pipeline → cost log.
    6. **Hallucination check** — LLM-judge runs on the answer + the
       retrieved sources. NOT_SUPPORTED triggers a SAFE_FILTERED_MESSAGE
       response with ``blocked_by="hallucination: ..."``.
    """
    # 1. LLM10 — rate limit (anchored CVE-2025-53773 cost-amplification).
    rl_reason = check_rate_limit(client_id)
    if rl_reason is not None:
        return safe_response(SAFE_BLOCKED_MESSAGE, blocked_by=rl_reason)

    # 2. Prompt injection (anchored OWASP LLM01:2025).
    pi_reason = detect_prompt_injection(request.question)
    if pi_reason is not None:
        return safe_response(SAFE_BLOCKED_MESSAGE, blocked_by=pi_reason)

    # 3. System-prompt leak (anchored CVE-2025-54135).
    spl_reason = detect_system_prompt_leak(request.question)
    if spl_reason is not None:
        return safe_response(SAFE_BLOCKED_MESSAGE, blocked_by=spl_reason)

    # 4. PII redaction — redact, pass through. ``cleaned`` is what the
    #    pipeline sees; the original is dropped on the floor.
    cleaned, pii_kinds = detect_pii(request.question)
    pii_reason: str | None = None
    if pii_kinds:
        pii_reason = f"pii_redacted: {','.join(pii_kinds)}"

    # 5. Dispatch.
    response = route_query(
        cleaned,
        top_k=request.top_k,
        client_id=client_id,
    )

    # 6. Hallucination check on the output.
    passed, halluc_reason = check_hallucination(response.answer, response.citations)
    if not passed:
        return safe_response(SAFE_FILTERED_MESSAGE, blocked_by=halluc_reason or "hallucination: judge flagged")

    # Stamp the PII reason (if any) onto the response so the operator
    # sees the redaction in the audit log even on the happy path.
    if pii_reason is not None:
        response.blocked_by = pii_reason

    # TODO(m20-exercise-4)-start
    # 7. Structured-output validation at the gateway boundary.
    #    ``QueryResponse`` carries the ``citations`` min_length=1 and
    #    ``confidence`` ∈ [0, 1] Pydantic ``Field`` constraints the
    #    planning doc (Skill Pair 9, exercise 4) pins. Re-validating
    #    the dumped response catches a contract violation introduced
    #    upstream — a validation failure here is a server-side bug,
    #    not a user error, so we return 502 (Bad Gateway), not 4xx.
    try:
        QueryResponse.model_validate(response.model_dump())
    except ValidationError as exc:
        first_error = exc.errors()[0]
        return JSONResponse(
            status_code=502,
            content={
                "detail": "output_validation_failed",
                "field": str(first_error.get("loc", ("unknown",))[0]),
            },
        )
    # TODO(m20-exercise-4)-end

    return response


@router.get(constants.HEALTH_ROUTE)
def health_endpoint() -> dict[str, str]:
    """Liveness probe — no dependencies, no I/O, just a 200 + status."""
    return {"status": "ok"}


__all__ = ["router", "QueryRequest", "query_endpoint", "health_endpoint"]
