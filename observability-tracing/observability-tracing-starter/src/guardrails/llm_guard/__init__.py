"""LLM Guard wrappers — DeBERTa prompt injection + Presidio PII (REQ-072, M20).

The capstone teaches the regex layer first because it is fast,
explainable, and grep-able. This subpackage is the ML layer that
catches what regex cannot enumerate. Both layers run; they are
complementary, not competing.

Module load is intentionally lazy — importing this package does not
download the DeBERTa weights or boot the Presidio NLP engine. The
scanners are constructed on first call to
:func:`src.guardrails.llm_guard.input_guards.detect_prompt_injection`
or :func:`src.guardrails.llm_guard.input_guards.detect_pii`. That keeps
the FastAPI startup cost low and lets the test suite mock the heavy
calls without touching the real scanner classes.
"""

from src.guardrails.llm_guard.input_guards import (
    detect_pii_layered,
    detect_prompt_injection_layered,
)

__all__ = ["detect_prompt_injection_layered", "detect_pii_layered"]
