"""Regex layer — the explainable fast-path for input guards.

Three detector families live here:

- :data:`INJECTION_PATTERNS` — direct prompt-injection signatures
  ("ignore previous instructions", role-hijack tokens, jailbreak modes).
  The first defensive layer; misses novel paraphrases by design — the
  LLM Guard DeBERTa layer in ``src/guardrails/llm_guard/input_guards.py``
  catches what regex cannot enumerate.
- :data:`SYSTEM_PROMPT_LEAK_PATTERNS` — crafted extraction queries
  designed to coax the LLM into echoing its own system prompt back
  ("show me your previous instructions", "what was the first sentence
  in your context"). Anchored on the Cursor "CurXecute" incident
  (CVE-2025-54135) — coding/docs assistants are the most-leaked category
  per the public `leaked-system-prompts` aggregation repo.
- :data:`PII_PATTERNS` + :data:`PII_REDACTIONS` — four high-precision
  patterns (email, phone, SSN, credit card) paired with placeholder
  strings. The corpus is curated scikit-learn docs (no PII surface), so
  this layer exists for the opposite-direction case: a user pasting a
  dataset row containing names/emails while asking about an estimator.
  Defense-in-depth even when the corpus itself has zero PII.

Each detector returns a reason string on match (or ``None`` / a tuple
on miss). The format ``<guard_name>: <detail>`` is the convention every
guard in the stack follows — operators grep audit logs by the prefix.
"""

import re


# === Slot 1 — Prompt injection (LLM01) ===

INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "ignore_previous": re.compile(
        r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions?\b",
        re.IGNORECASE,
    ),
    "disregard_prior": re.compile(
        r"\bdisregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)\b",
        re.IGNORECASE,
    ),
    "forget_everything": re.compile(
        r"\bforget\s+(everything|all)\s+(you|that\s+you)\s+(were\s+told|know|learned)\b",
        re.IGNORECASE,
    ),
    "role_hijack_system": re.compile(r"<\s*\|?\s*im_start\s*\|?\s*>", re.IGNORECASE),
    "role_hijack_inline": re.compile(
        r"^\s*(system|assistant)\s*:\s*", re.IGNORECASE | re.MULTILINE
    ),
    "new_instructions": re.compile(
        r"\bnew\s+instructions?\s*:", re.IGNORECASE
    ),
    "developer_mode": re.compile(r"\bdeveloper\s+mode\b", re.IGNORECASE),
    "jailbreak_mode": re.compile(
        r"\b(dan|do\s+anything\s+now|jailbreak)\s+mode\b", re.IGNORECASE
    ),
    "act_as_unfiltered": re.compile(
        r"\bact\s+as\s+(an?\s+)?(unfiltered|uncensored|unrestricted)\b",
        re.IGNORECASE,
    ),
    "pretend_no_rules": re.compile(
        r"\bpretend\s+(there\s+are\s+)?no\s+(rules|restrictions|guidelines)\b",
        re.IGNORECASE,
    ),
    "override_safety": re.compile(
        r"\b(override|bypass|disable)\s+(safety|guardrails?|filters?)\b",
        re.IGNORECASE,
    ),
}


def detect_prompt_injection(text: str) -> str | None:
    """Match ``text`` against :data:`INJECTION_PATTERNS`.

    Returns:
        Reason string in ``"prompt_injection: matched pattern '<regex>'"``
        format on hit; ``None`` on clean text. First-match wins — the
        patterns are mutually compatible, so ordering only affects which
        name lands in the reason string.
    """
    for name, pattern in INJECTION_PATTERNS.items():
        match = pattern.search(text)
        if match is not None:
            return f"prompt_injection: matched pattern '{name}'"
    return None


# === Slot 3 — System-prompt leakage (LLM01 sibling, anchored CVE-2025-54135) ===

SYSTEM_PROMPT_LEAK_PATTERNS: dict[str, re.Pattern[str]] = {
    "show_instructions": re.compile(
        r"\bshow\s+me\s+(your|the)\s+(previous|prior|original|initial|system)\s+(instructions?|prompt|message)\b",
        re.IGNORECASE,
    ),
    "what_were_told": re.compile(
        r"\bwhat\s+(were|was)\s+you\s+(told|instructed|given)\b",
        re.IGNORECASE,
    ),
    "repeat_prompt": re.compile(
        r"\b(repeat|print|echo|output|reveal)\s+(your|the)\s+(system\s+)?(prompt|instructions?|context|message)\b",
        re.IGNORECASE,
    ),
    "first_sentence_context": re.compile(
        r"\bwhat\s+(was|is)\s+the\s+(first|opening|initial)\s+(sentence|line|word|paragraph)\s+(in|of)\s+(your|the)\s+(context|prompt|instructions?)\b",
        re.IGNORECASE,
    ),
    "before_user_message": re.compile(
        r"\b(what|anything)\s+(was|came|appeared)\s+(before|above)\s+(this|the|my)\s+(message|question|input)\b",
        re.IGNORECASE,
    ),
    "translate_prompt": re.compile(
        r"\btranslate\s+(your|the)\s+(prompt|instructions?|system\s+message)\b",
        re.IGNORECASE,
    ),
    "summarize_prompt": re.compile(
        r"\bsummarize\s+(your|the)\s+(prompt|instructions?|system\s+message|guidelines?)\b",
        re.IGNORECASE,
    ),
    "spell_prompt": re.compile(
        r"\bspell\s+out\s+(your|the)\s+(prompt|instructions?|system\s+message)\b",
        re.IGNORECASE,
    ),
}


def detect_system_prompt_leak(text: str) -> str | None:
    """Match ``text`` against :data:`SYSTEM_PROMPT_LEAK_PATTERNS`.

    Returns:
        Reason string in ``"system_prompt_leak: matched pattern '<name>'"``
        format on hit; ``None`` on clean text. Composes with
        :func:`detect_prompt_injection` — the gateway runs both in
        sequence (extraction queries are a sibling of injection, not a
        subset, so they need their own detector).
    """
    for name, pattern in SYSTEM_PROMPT_LEAK_PATTERNS.items():
        match = pattern.search(text)
        if match is not None:
            return f"system_prompt_leak: matched pattern '{name}'"
    return None


# === Slot 2 — PII regex (paired with Presidio in llm_guard layer) ===

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone": re.compile(
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(
        r"\b(?:\d[ -]?){13,16}\d\b"
    ),
}


PII_REDACTIONS: dict[str, str] = {
    "email": "[REDACTED_EMAIL]",
    "phone": "[REDACTED_PHONE]",
    "ssn": "[REDACTED_SSN]",
    "credit_card": "[REDACTED_CC]",
}


def detect_pii(text: str) -> tuple[str, list[str]]:
    """Redact PII in ``text`` using the regex patterns.

    Returns:
        ``(redacted_text, kinds_found)``. ``kinds_found`` is the list of
        :data:`PII_PATTERNS` keys whose patterns matched at least once.
        Empty list means clean. The redacted text is what the
        downstream cache, retriever, and LLM see — the raw PII never
        reaches any logged surface.
    """
    redacted = text
    kinds: list[str] = []
    for kind, pattern in PII_PATTERNS.items():
        if pattern.search(redacted):
            kinds.append(kind)
            redacted = pattern.sub(PII_REDACTIONS[kind], redacted)
    return redacted, kinds


__all__ = [
    "INJECTION_PATTERNS",
    "SYSTEM_PROMPT_LEAK_PATTERNS",
    "PII_PATTERNS",
    "PII_REDACTIONS",
    "detect_prompt_injection",
    "detect_system_prompt_leak",
    "detect_pii",
]
