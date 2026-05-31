"""Blue/green re-ingest with golden-set eval gate (REQ-074, M24).

A migration is four moves:

1. Pick the **inactive** color (the opposite of whatever
   ``data/ACTIVE_COLLECTION`` currently names; bootstrap → blue).
2. Drop and rebuild that color from a pinned scikit-learn source tag.
   The bulk path is reused — ``scripts.load_data`` does the parse,
   chunk, embed, upsert work, but writing into the inactive color
   instead of the public alias.
3. Run a small ``recall@k`` gate against the golden set CSV
   (``data/golden_set.csv``) targeting the just-built color directly.
4. On pass: ``swap_alias`` atomically points the public alias at the
   new color. On fail: leave the alias where it was and (optionally)
   drop the failed color so the next attempt starts clean.

This is intentionally the simplest blue/green that teaches the
property — atomic cutover with a quality gate. A production system
would add traffic-shadowing, gradual ramp-up, longer eval sets, and
likely a human approval step at the swap. M24's exercise reasons about
those extensions; the runnable code is the minimum that demonstrates
why an atomic alias swap is safer than in-place re-ingest.
"""

from __future__ import annotations

import csv
import importlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from src import store
from src.ingestion.alias import (
    BLUE_NAME,
    DEFAULT_COLLECTION_NAME,
    GREEN_NAME,
    other_color,
    read_active_collection,
    swap_alias,
)

logger = logging.getLogger("ingestion.migrate")

GOLDEN_SET_PATH: Path = Path("data/golden_set.csv")
DEFAULT_RECALL_THRESHOLD: float = 0.70
DEFAULT_TOP_K: int = 5
# Use a subset of the full 30-question golden set for the gate.
# Migration eval is a fast confidence check; the long RAGAS sweep in
# REQ-068/M11 is the rigorous post-merge run.
DEFAULT_EVAL_SAMPLE_SIZE: int = 12


@dataclass(frozen=True)
class MigrationOutcome:
    """Return value of :func:`migrate_blue_green`.

    ``swapped`` is the load-bearing field; everything else exists so
    the CLI can print a human summary and tests can assert on the gate
    numbers without re-reading the eval CSV.
    """

    target_color: str
    swapped: bool
    recall_at_k: float
    threshold: float
    eval_n: int
    duration_seconds: float
    previous_color: str
    reason: str = ""


def _build_inactive_color(target_color: str, source_tag: str) -> int:
    """Bulk-ingest into ``target_color`` from ``source_tag``. Return chunk count.

    Reuses ``scripts.load_data`` helpers so the embedding cache stays
    shared with the alias-path ``make load-data`` flow — most chunks
    on a same-version re-ingest are cache hits.
    """
    load_data = importlib.import_module("scripts.load_data")

    sha = load_data.ensure_repo_cache(load_data.REPO_CACHE_DIR, source_tag)
    logger.info("scikit-learn@%s → %s", source_tag, sha[:12])

    chunks: list[dict] = []
    seen: set[str] = set()
    from src import corpus as corpus_mod

    for section in corpus_mod.load_corpus(load_data.REPO_CACHE_DIR, sha):
        for chunk in load_data.chunk_section(section):
            if chunk["doc_id"] in seen:
                continue
            seen.add(chunk["doc_id"])
            chunks.append(chunk)

    if not chunks:
        raise RuntimeError(
            f"corpus@{source_tag} produced 0 chunks — nothing to migrate"
        )

    client = load_data._make_openai_client()
    cache = load_data.load_embedding_cache(load_data.EMBEDDING_CACHE_PATH)
    embeddings = load_data.embed_missing(client, chunks, cache)

    # Bypass the alias by getting the named color collection directly.
    # Drop-and-rebuild gives a clean slate: prior failed migration
    # attempts won't leave stale rows in the target collection.
    client_chroma = store._client()
    try:
        client_chroma.delete_collection(name=target_color)
    except Exception:  # noqa: BLE001 — chroma raises NotFoundError or similar
        pass
    collection = store.get_collection(target_color)
    load_data.upsert_chunks(collection, chunks, embeddings)

    return len(chunks)


def _load_golden_subset(path: Path, sample_size: int) -> list[dict]:
    """Return the first ``sample_size`` rows of the golden CSV.

    Subset order is the CSV's row order, which REQ-063 balanced across
    question-type buckets — taking the head is a fair sample without
    extra balancing logic.
    """
    if not path.exists():
        raise FileNotFoundError(f"golden set not found at {path}")
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
            if len(rows) >= sample_size:
                break
    return rows


def recall_at_k(
    collection_name: str,
    golden_rows: list[dict],
    top_k: int = DEFAULT_TOP_K,
) -> float:
    """Hit-any recall@k against ``collection_name`` for ``golden_rows``.

    A row hits when any of the pipe-separated section prefixes in
    ``expected_doc_ids`` is a prefix of any retrieved chunk's id. The
    metric is intentionally lenient — section-prefix matching tolerates
    REQ-065's ``.p0`` / ``.p1`` chunk splits without per-row tweaks.
    """
    if not golden_rows:
        return 0.0

    from src import embedder

    hits = 0
    for row in golden_rows:
        question = row["question"]
        expected_raw = row.get("expected_doc_ids", "")
        prefixes = [p.strip() for p in expected_raw.split("|") if p.strip()]
        if not prefixes:
            continue
        query_vec = embedder.embed_query(question)
        result = store.get_collection(collection_name).query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents"],
        )
        retrieved_ids = result["ids"][0]
        if any(rid.startswith(p) for rid in retrieved_ids for p in prefixes):
            hits += 1
    return hits / len(golden_rows)


def migrate_blue_green(
    source_tag: str | None = None,
    threshold: float = DEFAULT_RECALL_THRESHOLD,
    eval_sample_size: int = DEFAULT_EVAL_SAMPLE_SIZE,
    drop_failed: bool = True,
) -> MigrationOutcome:
    """End-to-end blue/green migration. Returns a :class:`MigrationOutcome`.

    Args:
        source_tag: scikit-learn git tag to ingest. ``None`` defaults to
            ``corpus.SCIKIT_LEARN_TAG`` — the same-version re-build form
            of blue/green that the exercise uses. Pass an explicit tag
            (``"1.6.0"``) for an actual version-upgrade migration.
        threshold: recall@k floor below which the swap is refused.
        eval_sample_size: golden-set rows to evaluate; the head of the
            CSV is used so the type-balance from REQ-063 is preserved.
        drop_failed: on gate failure, delete the freshly-built color so
            the next attempt starts clean. Set ``False`` for forensics.
    """
    from src import corpus as corpus_mod

    started = time.monotonic()
    previous = read_active_collection()
    target = other_color(previous)
    if source_tag is None:
        source_tag = corpus_mod.SCIKIT_LEARN_TAG

    logger.info(
        "Building %s from scikit-learn@%s (previous active=%s)",
        target,
        source_tag,
        previous,
    )
    _build_inactive_color(target, source_tag)

    golden_rows = _load_golden_subset(GOLDEN_SET_PATH, eval_sample_size)
    score = recall_at_k(target, golden_rows, top_k=DEFAULT_TOP_K)
    logger.info(
        "Gate: recall@%d=%.3f vs threshold=%.3f on n=%d rows",
        DEFAULT_TOP_K,
        score,
        threshold,
        len(golden_rows),
    )

    elapsed = time.monotonic() - started
    if score < threshold:
        reason = (
            f"recall@{DEFAULT_TOP_K}={score:.3f} below threshold "
            f"{threshold:.3f}; alias unchanged"
        )
        if drop_failed:
            try:
                store._client().delete_collection(name=target)
                logger.info("Dropped failed color %s", target)
            except Exception:  # noqa: BLE001
                logger.warning("Could not drop failed color %s", target)
        return MigrationOutcome(
            target_color=target,
            swapped=False,
            recall_at_k=score,
            threshold=threshold,
            eval_n=len(golden_rows),
            duration_seconds=elapsed,
            previous_color=previous,
            reason=reason,
        )

    swap_alias(target)
    logger.info("Swapped alias: %s → %s", previous, target)
    return MigrationOutcome(
        target_color=target,
        swapped=True,
        recall_at_k=score,
        threshold=threshold,
        eval_n=len(golden_rows),
        duration_seconds=elapsed,
        previous_color=previous,
        reason="",
    )


__all__ = [
    "BLUE_NAME",
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_EVAL_SAMPLE_SIZE",
    "DEFAULT_RECALL_THRESHOLD",
    "DEFAULT_TOP_K",
    "GOLDEN_SET_PATH",
    "GREEN_NAME",
    "MigrationOutcome",
    "migrate_blue_green",
    "recall_at_k",
]
