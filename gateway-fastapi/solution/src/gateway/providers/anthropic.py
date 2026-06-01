"""Anthropic Messages API adapter — converts gateway I/O to Anthropic's shape.

Demonstrates the multi-provider abstraction pattern the gateway concept module
named. The starter routes only OpenAI in production; this adapter is the
pattern, not the credentials. A live integration would import the
``anthropic`` SDK and replace the stub at ``_call_anthropic`` below with
``anthropic.Anthropic(api_key=...).messages.create(...)``.

Three shape conversions are pinned here because they are the gotchas every
multi-provider adapter eventually re-discovers:

1. **System prompt placement** — Anthropic takes ``system`` as a top-level
   argument; OpenAI mixes it into ``messages`` as the first turn.
2. **Response shape** — Anthropic returns a list of typed content blocks;
   OpenAI returns a single message string.
3. **Token-count field names** — Anthropic uses ``input_tokens`` /
   ``output_tokens``; OpenAI uses ``prompt_tokens`` / ``completion_tokens``.
   A less-careful adapter would pass ``response.usage`` through and the
   cost computation would silently see zero tokens.
"""

# TODO(m18-ex3): author the Anthropic adapter — stubbed _call_anthropic + generate() with the same (answer, TokenUsage, cost) contract as src.generator.generate

from src.models import TokenUsage
from src.pricing import compute_cost


def _call_anthropic(
    model: str, system_prompt: str, question: str, max_tokens: int = 1024
) -> dict:
    """STUB — replace with ``anthropic.Anthropic(api_key=...).messages.create(...)``.

    Returns the Anthropic Messages API response shape so the calling code can
    be tested without an Anthropic key. The real response shape from
    ``anthropic.messages.create()`` has the same ``content`` list of text
    blocks and the same ``usage`` dict with ``input_tokens`` and
    ``output_tokens``.
    """
    return {
        "id": "msg_stub_0001",
        "model": model,
        "content": [
            {"type": "text", "text": f"[stub] answer for: {question[:40]}"}
        ],
        "usage": {"input_tokens": 410, "output_tokens": 35},
    }


def generate(
    question: str, system_prompt: str, model: str
) -> tuple[str, TokenUsage, float]:
    """Provider-shaped generator with the same return contract as ``src.generator.generate``.

    Two shape conversions happen here. On the response side, Anthropic
    returns a list of content blocks rather than a single message string, so
    we concatenate the text-typed blocks. Token usage uses different field
    names (``input_tokens`` / ``output_tokens`` vs OpenAI's
    ``prompt_tokens`` / ``completion_tokens``) and we remap.
    """
    response = _call_anthropic(model, system_prompt, question)
    answer = "".join(
        block["text"] for block in response["content"] if block["type"] == "text"
    )
    usage = TokenUsage(
        prompt_tokens=response["usage"]["input_tokens"],
        completion_tokens=response["usage"]["output_tokens"],
    )
    cost = compute_cost(model, usage)
    return answer, usage, cost


__all__ = ["generate"]
