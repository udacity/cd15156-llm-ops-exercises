"""LLM Guard layer — DeBERTa + Presidio for the input stack (Module 20).

Two functions, mirroring the regex layer's shape:

- :func:`detect_prompt_injection_layered` — regex short-circuit first,
  then DeBERTa via ``llm_guard.input_scanners.PromptInjection``. Returns
  the regex reason when the fast-path fires, the DeBERTa reason
  otherwise.
- :func:`detect_pii_layered` — regex first (high-precision exact
  matchers), then Presidio NER for the entity categories regex cannot
  enumerate (names, locations, organisations). ``kinds_found`` is the
  union of both layers.

Two library-specific patches:

1. ``ALL_SUPPORTED_LANGUAGES = ["en"]`` is set BEFORE the Anonymize
   import so Presidio does not try to download ``zh_core_web_sm`` at
   first run and pull in ``spacy-pkuseg``'s compile step (fails on the
   workspace image — `llm-guard` issue 337).
2. The Presidio recognizer config is overridden to
   ``dslim/bert-base-NER`` (MIT) rather than upstream's default
   ``Isotonic/deberta-v3-base_finetuned_ai4privacy_v2`` (CC-BY-NC-4.0).
   The license override matters for any learner who extends the
   starter commercially.
"""

from __future__ import annotations

from threading import Lock
from typing import Any

from src.guardrails.input_guards import detect_pii, detect_prompt_injection


_INIT_LOCK = Lock()
_PROMPT_INJECTION_SCANNER: Any = None
_ANONYMIZE_SCANNER: Any = None
_PII_VAULT: Any = None


def _ensure_scanners() -> None:
    """Lazy-load LLM Guard scanners on first call.

    The imports are inside the function body so importing this module
    is cheap. The DeBERTa weights (~250 MB) and the Presidio NLP engine
    only load when the real scanners are needed — tests mock at this
    seam and never trigger the download.
    """
    global _PROMPT_INJECTION_SCANNER, _ANONYMIZE_SCANNER, _PII_VAULT

    if _PROMPT_INJECTION_SCANNER is not None and _ANONYMIZE_SCANNER is not None:
        return

    with _INIT_LOCK:
        if _PROMPT_INJECTION_SCANNER is not None and _ANONYMIZE_SCANNER is not None:
            return

        # llm-guard issue 337 workaround — must mutate ALL_SUPPORTED_LANGUAGES
        # BEFORE importing Anonymize so Presidio's first-run download path
        # only reaches for en_core_web_sm. Without the patch, the import
        # path tries zh_core_web_sm and the spacy-pkuseg wheel build fails
        # on the workspace image.
        from llm_guard.input_scanners import anonymize as _anonymize_mod

        _anonymize_mod.ALL_SUPPORTED_LANGUAGES = ["en"]
        assert _anonymize_mod.ALL_SUPPORTED_LANGUAGES == ["en"], (
            "llm-guard layout changed — re-evaluate the issue-337 patch"
        )

        from llm_guard.input_scanners import Anonymize, PromptInjection
        from llm_guard.input_scanners.anonymize_helpers import (
            BERT_BASE_NER_CONF,
        )
        from llm_guard.vault import Vault

        _PROMPT_INJECTION_SCANNER = PromptInjection()
        _PII_VAULT = Vault()
        _ANONYMIZE_SCANNER = Anonymize(
            vault=_PII_VAULT,
            recognizer_conf=BERT_BASE_NER_CONF,
        )


def detect_prompt_injection_layered(text: str) -> str | None:
    """Regex short-circuit then DeBERTa.

    The fast-path is :func:`src.guardrails.input_guards.detect_prompt_injection`.
    If regex fires, this function returns its reason without touching
    DeBERTa — that avoids the model call on the cheapest attacks (which
    is the entire point of running regex first).

    On a regex miss, the LLM Guard ``PromptInjection`` scanner runs.
    The scanner's ``scan`` method returns ``(sanitized_text, is_valid,
    risk_score)``. We return a reason string with the risk score on
    block, ``None`` on clean.
    """
    regex_reason = detect_prompt_injection(text)
    if regex_reason is not None:
        return regex_reason

    _ensure_scanners()
    _sanitized, is_valid, risk_score = _PROMPT_INJECTION_SCANNER.scan(text)
    if not is_valid:
        return f"prompt_injection: risk_score={risk_score:.3f}"
    return None


def detect_pii_layered(text: str) -> tuple[str, list[str]]:
    """Regex layer first, then Presidio NER.

    Both layers always run — regex catches structural PII (email,
    phone, SSN, credit card) with near-zero false-positive rate;
    Presidio catches everything regex cannot enumerate (names,
    locations, organisations). ``kinds_found`` is the union.
    """
    redacted, regex_kinds = detect_pii(text)

    _ensure_scanners()
    sanitized, is_valid, _risk = _ANONYMIZE_SCANNER.scan(redacted)
    if not is_valid:
        # Presidio found at least one entity not caught by regex. The
        # scanner returns the redacted text directly; we treat the
        # named scanner as the "presidio_pii" kind.
        kinds = list(regex_kinds)
        if "presidio_pii" not in kinds:
            kinds.append("presidio_pii")
        return sanitized, kinds

    return redacted, regex_kinds


__all__ = ["detect_prompt_injection_layered", "detect_pii_layered"]
