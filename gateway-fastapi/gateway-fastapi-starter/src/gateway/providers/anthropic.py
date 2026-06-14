"""Anthropic Messages API adapter — converts gateway I/O to Anthropic's shape.

This adapter shows the multi-provider routing pattern. The stub at
``_call_anthropic`` below returns a canned Anthropic-shaped response so
you can test routing without a second key — point ``provider="anthropic"``
at it and the rest of the gateway treats it like any other backend. To go
live, you would import the ``anthropic`` SDK and replace the stub body with
``anthropic.Anthropic(api_key=...).messages.create(...)``; nothing else
changes.

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
raise NotImplementedError("TODO(m18-ex3)")
