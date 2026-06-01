"""Input + output guardrails for the ScikitDocs gateway (Module 20).

Four guardrail slots, matched 1:1 to the OWASP LLM Top 10 categories
the module exercises:

1. **Prompt injection** (LLM01) — regex fast-path + DeBERTa classifier.
   Layered: regex short-circuits the cheap attacks, DeBERTa generalises
   to novel phrasings.
2. **PII detection** (LLM02 retention slot) — Presidio NER + regex
   patterns. The corpus is curated scikit-learn docs (zero PII surface),
   but defense-in-depth catches the opposite direction: a user pasting
   a dataset row with names/emails while asking about an estimator.
3. **System-prompt leakage** (LLM01 sibling slot) — regex patterns for
   crafted extraction queries ("show me previous instructions", "what
   was the first sentence in your context", etc.). Anchored on the
   Cursor "CurXecute" incident (CVE-2025-54135).
4. **Unbounded consumption** (LLM10) — token-bucket request limiter
   plus a per-request output-token cap. Anchored on the Copilot RCE /
   cost-amplification incident (CVE-2025-53773).

The LLM-as-judge hallucination check (``src.guardrails.llm_judge``) sits
beside slots 1-4 as the output-side correctness layer. Exercise 2
calibrates its threshold against scikit-learn API-correctness examples.
"""

from src.guardrails.input_guards import (
    INJECTION_PATTERNS,
    PII_PATTERNS,
    PII_REDACTIONS,
    SYSTEM_PROMPT_LEAK_PATTERNS,
    detect_pii,
    detect_prompt_injection,
    detect_system_prompt_leak,
)
from src.guardrails.rate_limit import (
    MAX_OUTPUT_TOKENS,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    check_rate_limit,
    reset_rate_limit_state,
)
from src.guardrails.wrapper import (
    SAFE_BLOCKED_MESSAGE,
    SAFE_FILTERED_MESSAGE,
    safe_response,
)

__all__ = [
    "INJECTION_PATTERNS",
    "PII_PATTERNS",
    "PII_REDACTIONS",
    "SYSTEM_PROMPT_LEAK_PATTERNS",
    "detect_pii",
    "detect_prompt_injection",
    "detect_system_prompt_leak",
    "check_rate_limit",
    "reset_rate_limit_state",
    "MAX_OUTPUT_TOKENS",
    "RATE_LIMIT_REQUESTS",
    "RATE_LIMIT_WINDOW_SECONDS",
    "SAFE_BLOCKED_MESSAGE",
    "SAFE_FILTERED_MESSAGE",
    "safe_response",
]
