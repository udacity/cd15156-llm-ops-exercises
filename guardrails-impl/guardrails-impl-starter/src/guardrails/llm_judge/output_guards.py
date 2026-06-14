"""LLM-as-judge hallucination check.

A single function — :func:`check_hallucination` — that takes a generated
answer and the retrieved source chunks, asks gpt-4o-mini whether every
factual claim in the answer is supported by the chunks, and returns
``(passed, reason)``. JSON-mode response contract. Fail-open on any
error (network blip, JSON-decode error, missing field) so a transient
failure cannot block a grounded answer.

Cost rows are written to ``data/cost_log.jsonl`` via
:func:`src.cost.tracker.log_request` with ``query_type="hallucination_check"``
so the cost dashboard surfaces them in a separate bucket from real
answers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader, select_autoescape
from openai import OpenAI

from src import constants
from src.config import settings
from src.cost.tracker import log_request
from src.models import Source, TokenUsage

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


_PROMPT_ROOT = Path(__file__).resolve().parents[3] / "prompts"
_JINJA_ENV = Environment(
    loader=FileSystemLoader(str(_PROMPT_ROOT)),
    autoescape=select_autoescape(disabled_extensions=("j2",)),
    keep_trailing_newline=True,
)


_client = OpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url or None,
)


def _render(answer: str, sources: list[Source]) -> str:
    """Render the judge prompt with the answer and retrieved sources."""
    template = _JINJA_ENV.get_template("judge.j2")
    context_blocks = "\n\n".join(
        f"[{s.doc_id}]\n{s.chunk_text}" for s in sources
    )
    return template.render(answer=answer, source=context_blocks)


def check_hallucination(
    answer: str, sources: list[Source]
) -> tuple[bool, str | None]:
    """Score whether ``answer`` is supported by ``sources``.

    Args:
        answer: The generator's answer string.
        sources: The retrieved chunks the generator used.

    Returns:
        ``(True, None)`` if the judge says SUPPORTED or if any error
        path triggers (fail-open). ``(False, "hallucination: <reason>")``
        if the judge says NOT_SUPPORTED. The block path is the only one
        that returns a non-``None`` reason — the caller wraps the reason
        in :func:`src.guardrails.wrapper.safe_response`.
    """
    if not answer.strip():
        return True, None
    if not sources:
        return False, "hallucination: no sources retrieved to support answer"

    try:
        prompt = _render(answer, sources)
        completion = _client.chat.completions.create(
            model=constants.MODEL_SIMPLE,
            messages=[{"role": "user", "content": prompt}],
            temperature=constants.JUDGE_TEMPERATURE,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content or "{}"
        usage = completion.usage
        if usage is not None:
            log_request(
                model=constants.MODEL_SIMPLE,
                usage=TokenUsage(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                ),
                cost_usd=0.0,
                query_type="hallucination_check",
            )
        payload = json.loads(raw)
    except Exception as exc:  # network, JSON, key missing — all fail-open
        logger.warning("hallucination check failed open: %s", exc)
        return True, None

    verdict = (payload.get("verdict") or "").strip().upper()
    reason = (payload.get("reason") or "").strip()
    if verdict == "NOT_SUPPORTED":
        detail = reason or "judge marked answer not supported by sources"
        return False, f"hallucination: {detail}"
    return True, None


__all__ = ["check_hallucination"]
