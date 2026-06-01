"""OpenAI generation + system-prompt rendering (Module 03).

Renders ``prompts/docbot_system.j2`` with the retrieved chunks as context
and calls OpenAI chat completions. Frozen contract documented in
``INTERFACES.md``.

Two design choices worth naming:

- ``autoescape=False`` because these are plaintext prompts going to the
  LLM, not HTML — escaping ``&`` or ``{`` would corrupt the message.
- ``keep_trailing_newline=True`` because Jinja strips the final newline
  by default and removing it can shift tokenization on some models.

``cost_usd`` is computed by ``src.pricing.compute_cost`` (added by
the initial scaffolding / Module 13). The signature still matches ``INTERFACES.md`` —
Module 13 wired the real cost in without changing the return shape.
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

# TODO(m18-ex2): import tenacity + openai exception types for the retry wrapper
from openai import OpenAI

from src import constants
from src.config import settings
from src.models import Source, TokenUsage
from src.pricing import compute_cost

# ``parents[1]`` lands on the starter root (src/ is one level under it).
# The capstone uses ``parents[2]`` because its generator lives in
# src/rag/. Don't generalise this — it should be obvious which directory
# the templates live in from the file path alone.
_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
_env = Environment(
    loader=FileSystemLoader(_PROMPTS_DIR),
    keep_trailing_newline=True,
    autoescape=False,
)


def render_system_prompt(sources: list[Source]) -> str:
    """Render ``docbot_system.j2`` with retrieved chunks as context.

    Args:
        sources: List of retrieved chunks (from ``store.query``).

    Returns:
        The fully-rendered system prompt string.
    """
    template = _env.get_template("docbot_system.j2")
    contexts = "\n\n---\n\n".join(s.chunk_text for s in sources)
    return template.render(contexts=contexts)


# TODO(m18-ex2): add _is_retryable + @retry-decorated _call_chat_completions helper
def generate(
    question: str, sources: list[Source], model: str
) -> tuple[str, TokenUsage, float]:
    """Call OpenAI chat completions and return (answer, usage, cost_usd).

    Args:
        question: User's question.
        sources:  Retrieved chunks for the system prompt.
        model:    Model name (callers default to
                  ``constants.MODEL_COMPLEX``).

    Returns:
        ``(answer, TokenUsage, cost_usd)``. ``cost_usd`` comes from
        ``src.pricing.compute_cost`` (wired in by Module 13).
    """
    client = OpenAI(base_url=settings.openai_base_url or None)
    system_prompt = render_system_prompt(sources)
    # TODO(m18-ex2): route the bare client.chat.completions.create through _call_chat_completions
    response = client.chat.completions.create(
        model=model,
        temperature=constants.GENERATION_TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    answer = response.choices[0].message.content or ""
    usage = TokenUsage(
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
    )
    cost_usd = compute_cost(model, usage)
    return answer, usage, cost_usd
