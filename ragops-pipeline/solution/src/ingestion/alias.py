"""Blue/green alias file management.

The ``scikit_docs`` collection name is a *public alias* that resolves to
one of two real Chroma collections: ``scikit_docs_blue`` or
``scikit_docs_green``. Which one is "active" — the target of every
``store.get_collection("scikit_docs")`` call — is recorded as one line in
``data/ACTIVE_COLLECTION``.

The file is intentionally trivial. The atomic swap is a
``write-tmp + os.replace`` on the same filesystem, which POSIX
guarantees as an atomic rename — there is no half-written window during
which a concurrent reader could see a partial value. That property is
the load-bearing reason this file exists rather than the alias being a
``constants.py`` value.

Before any migration has run there is no ``ACTIVE_COLLECTION`` file and
``read_active_collection`` returns the literal ``scikit_docs`` so the
original single-collection behaviour is preserved for any caller that
imports :func:`src.store.get_collection`.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_COLLECTION_NAME: str = "scikit_docs"
BLUE_NAME: str = "scikit_docs_blue"
GREEN_NAME: str = "scikit_docs_green"

ACTIVE_COLLECTION_PATH: Path = Path("data/ACTIVE_COLLECTION")

_VALID_COLORS: frozenset[str] = frozenset({BLUE_NAME, GREEN_NAME})


def read_active_collection(path: Path = ACTIVE_COLLECTION_PATH) -> str:
    """Return the active color name, or ``DEFAULT_COLLECTION_NAME`` if unset.

    Falling back to the alias name (rather than guessing blue/green)
    means a fresh checkout of the starter behaves as it did before the alias migration — ``get_collection("scikit_docs")`` returns the original
    single-collection Chroma store, and no migration is implied.
    """
    if not path.exists():
        return DEFAULT_COLLECTION_NAME
    value = path.read_text(encoding="utf-8").strip()
    return value or DEFAULT_COLLECTION_NAME


def swap_alias(target: str, path: Path = ACTIVE_COLLECTION_PATH) -> str:
    """Atomically point the alias at ``target``; return the previous value.

    ``target`` must name one of the two colors; passing the alias name
    itself or an arbitrary string is a programming error and raises
    ``ValueError`` before any filesystem mutation happens.

    The write goes to ``<path>.tmp`` on the same directory, then
    ``os.replace`` performs the atomic rename. Readers calling
    :func:`read_active_collection` concurrently see either the previous
    value or the new one — never an empty or partial file.
    """
    if target not in _VALID_COLORS:
        raise ValueError(
            f"alias target must be one of {sorted(_VALID_COLORS)!r}; got {target!r}"
        )
    previous = read_active_collection(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(target + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return previous


def other_color(name: str) -> str:
    """Return the opposite color name. Used to pick the inactive slot.

    Pass either a color name (returns the other color) or the alias /
    any unset value (defaults to ``BLUE_NAME`` — the bootstrap target
    for the first migration on a previously-unaliased starter).
    """
    if name == BLUE_NAME:
        return GREEN_NAME
    if name == GREEN_NAME:
        return BLUE_NAME
    return BLUE_NAME
