"""Deprecated-API faithfulness sub-metric (REQ-068, M11).

Scores whether any scikit-learn symbol cited in a generated answer is
deprecated or removed as of the pinned corpus version. The check is a
faithfulness-style fact-checker scoped to library-API correctness — a
generator can produce text that RAGAS rates as faithful (every claim
supported by retrieved context) while still recommending an API that
was removed three versions ago. This sub-metric catches that surface.

The allow-list is intentionally small. Each entry names a symbol the
official scikit-learn 1.5 release notes flag as removed or deprecated,
plus the replacement and the version the change landed in. Extending
the list is a small edit — add the symbol, the deprecation version,
and the recommended replacement. Production teams would generate this
from the scikit-learn release notes programmatically; the static list
keeps the M11 exercise self-contained.

CORPUS_VERSION pin: scikit-learn 1.5 (data/CORPUS_VERSION). Re-pin
when the corpus is re-ingested at a newer version.
"""

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class DeprecatedApi:
    symbol: str
    deprecated_in: str
    removed_in: str | None
    replacement: str | None
    note: str


# Anchored to scikit-learn 1.5 (the corpus version pinned by REQ-062).
# Every entry traces to the scikit-learn release notes at
# https://scikit-learn.org/stable/whats_new.html.
DEPRECATED_APIS: tuple[DeprecatedApi, ...] = (
    DeprecatedApi(
        symbol="sklearn.preprocessing.Imputer",
        deprecated_in="0.20",
        removed_in="0.22",
        replacement="sklearn.impute.SimpleImputer",
        note="Removed in 0.22; replaced by the sklearn.impute submodule.",
    ),
    DeprecatedApi(
        symbol="sklearn.cross_validation",
        deprecated_in="0.18",
        removed_in="0.20",
        replacement="sklearn.model_selection",
        note="Whole submodule removed in 0.20.",
    ),
    DeprecatedApi(
        symbol="sklearn.grid_search",
        deprecated_in="0.18",
        removed_in="0.20",
        replacement="sklearn.model_selection",
        note="Whole submodule removed in 0.20 alongside cross_validation.",
    ),
    DeprecatedApi(
        symbol="sklearn.learning_curve",
        deprecated_in="0.18",
        removed_in="0.20",
        replacement="sklearn.model_selection.learning_curve",
        note="Module relocated; the bare module path was removed in 0.20.",
    ),
    DeprecatedApi(
        symbol="sklearn.externals.joblib",
        deprecated_in="0.21",
        removed_in="0.23",
        replacement="joblib",
        note="Re-export removed; install the standalone joblib package.",
    ),
    DeprecatedApi(
        symbol="sklearn.utils.testing",
        deprecated_in="0.22",
        removed_in="0.24",
        replacement="sklearn.utils._testing",
        note="Public testing utilities renamed to a private path in 0.24.",
    ),
    DeprecatedApi(
        symbol="normalize=True",
        deprecated_in="1.0",
        removed_in="1.2",
        replacement="StandardScaler in a Pipeline",
        note="`normalize=True` argument removed from linear models in 1.2; use a preprocessing step instead.",
    ),
    DeprecatedApi(
        symbol="DecisionTreeClassifier(presort=...)",
        deprecated_in="0.22",
        removed_in="0.24",
        replacement="(remove argument; presort no longer affects fit)",
        note="`presort` argument removed in 0.24.",
    ),
)


# Pre-compute regex patterns once. The patterns are deliberately
# permissive — we accept the symbol with or without trailing parens
# or attribute access, because RAG answers cite both forms.
_PATTERNS: tuple[tuple[re.Pattern[str], DeprecatedApi], ...] = tuple(
    (re.compile(re.escape(api.symbol), re.IGNORECASE), api) for api in DEPRECATED_APIS
)


def find_deprecated_citations(answer: str) -> list[DeprecatedApi]:
    """Return every deprecated API mentioned in ``answer`` (de-duplicated, ordered)."""
    seen: list[DeprecatedApi] = []
    for pattern, api in _PATTERNS:
        if pattern.search(answer) and api not in seen:
            seen.append(api)
    return seen


def score_deprecated_apis(answer: str) -> float:
    """1.0 if no deprecated APIs are cited, 0.0 if any are.

    Binary by design — a single deprecated symbol in an answer is a
    correctness failure regardless of how many other symbols are
    correct. The per-row JSON output carries the offending symbol
    names alongside the score so a learner running the diagnostic
    loop sees which API tripped the metric.
    """
    return 1.0 if not find_deprecated_citations(answer) else 0.0


def aggregate(scores: Iterable[float]) -> float:
    """Mean of per-row scores; ``0.0`` if the iterable is empty."""
    values = list(scores)
    return sum(values) / len(values) if values else 0.0


__all__ = [
    "DEPRECATED_APIS",
    "DeprecatedApi",
    "aggregate",
    "find_deprecated_citations",
    "score_deprecated_apis",
]
