"""LLM self-classifier for the tiered router (REQ-071, M18).

gpt-4o-mini reads the user's question and returns ``{"classification":
"simple" | "complex", "reasoning": "..."}`` via OpenAI's JSON mode.
The two-tier output keeps the starter minimal — Exercise 1 of M18 walks
the four-file edit to add a third ``premium`` tier without changing the
gateway scaffolding.

The fall-through at :func:`classify` defaults bad JSON or unexpected
labels to ``complex``. The reasoning is conservative: when the
classifier returns garbage, pay a bit more for the better model rather
than silently routing a hard question to the cheap path. M18 Slide 9
("Common pitfalls") names this as the standard default-on-failure
pattern for tier classifiers.
"""

import json
from pathlib import Path
from typing import Literal

from jinja2 import Environment, FileSystemLoader
from openai import OpenAI

from src import constants
from src.config import settings

QueryType = Literal["simple", "complex", "premium"]
_VALID_LABELS: tuple[str, ...] = ("simple", "complex", "premium")


_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
_env = Environment(
    loader=FileSystemLoader(_PROMPTS_DIR),
    keep_trailing_newline=True,
    autoescape=False,
)


def _render_classifier_prompt() -> str:
    """Render ``prompts/classifier.j2`` — no variables, just the rubric."""
    return _env.get_template("classifier.j2").render()


def classify(question: str, model: str | None = None) -> QueryType:
    """Classify ``question`` as ``simple`` or ``complex``.

    Args:
        question: The user's question, verbatim from the request body.
        model: Override the classifier model. Defaults to
            ``constants.MODEL_SIMPLE`` (gpt-4o-mini) — classification is
            itself a simple-tier task so paying gpt-4o rates here would
            waste the routing decision the classifier exists to enable.

    Returns:
        ``"simple"`` or ``"complex"``. Bad JSON or unexpected labels
        fall through to ``"complex"`` — see the module docstring for the
        rationale.
    """
    client = OpenAI(base_url=settings.openai_base_url or None)
    chosen_model = model or constants.MODEL_SIMPLE

    response = client.chat.completions.create(
        model=chosen_model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _render_classifier_prompt()},
            {"role": "user", "content": question},
        ],
    )
    raw = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return "complex"

    label = parsed.get("classification")
    if label not in _VALID_LABELS:
        return "complex"
    return label  # type: ignore[return-value]


__all__ = ["QueryType", "classify"]
