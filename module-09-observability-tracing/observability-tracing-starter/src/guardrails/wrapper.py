"""Safe-response helper used by every guardrail block path (Module 20).

When any guard fires, the gateway returns a 200 OK with a generic
refusal message in ``answer``, empty ``sources``, ``confidence=0.0``,
and the reason string in ``blocked_by``. HTTP-200-on-block is the
deliberate contract: an operator can grep audit logs by ``blocked_by``
without filtering on status codes, and the response shape stays
identical for clients (no need for a parallel error-handling branch).
"""

from src import constants
from src.models import QueryResponse, TokenUsage


SAFE_BLOCKED_MESSAGE: str = (
    "I can't help with that request. Please ask about the scikit-learn "
    "documentation surface this assistant covers."
)
"""Used when an input guard fires (prompt injection / system-prompt leak /
rate limit). The phrasing avoids confirming what was detected — an
attacker probing the guard surface gets the same refusal text every time.
"""


SAFE_FILTERED_MESSAGE: str = (
    "The retrieved sources did not support a confident answer to that "
    "question. Try rephrasing or asking about a different scikit-learn API."
)
"""Used when an output guard fires (hallucination / off-topic). Distinct
from ``SAFE_BLOCKED_MESSAGE`` so operators can tell input-side blocks
from output-side filters in the audit log without parsing ``blocked_by``.
"""


def safe_response(message: str, *, blocked_by: str) -> QueryResponse:
    """Construct the canonical block-response shape.

    Args:
        message: The user-facing refusal string. Pass
            ``SAFE_BLOCKED_MESSAGE`` for input-side blocks and
            ``SAFE_FILTERED_MESSAGE`` for output-side filters.
        blocked_by: Reason string in ``<guard_name>: <detail>`` format
            (e.g. ``"prompt_injection: matched pattern '\\\\bignore'"``).
            Surfaced in the response and recorded in traces.

    Returns:
        A :class:`QueryResponse` with empty sources, zero confidence,
        and the model field set to the cheaper tier — the actual model
        was never called.
    """
    return QueryResponse(
        answer=message,
        sources=[],
        confidence=0.0,
        model=constants.MODEL_SIMPLE,
        tokens=TokenUsage(prompt_tokens=0, completion_tokens=0),
        cost_usd=0.0,
        cached=False,
        blocked_by=blocked_by,
    )


__all__ = ["SAFE_BLOCKED_MESSAGE", "SAFE_FILTERED_MESSAGE", "safe_response"]
