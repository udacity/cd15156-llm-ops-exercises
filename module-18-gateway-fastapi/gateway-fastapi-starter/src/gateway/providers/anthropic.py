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
raise NotImplementedError("TODO(m18-ex3)")
