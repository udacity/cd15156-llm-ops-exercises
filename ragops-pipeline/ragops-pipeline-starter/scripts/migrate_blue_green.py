"""CLI entry for the blue/green corpus migration (Module 24).

Usage:
    uv run python scripts/migrate_blue_green.py
    uv run python scripts/migrate_blue_green.py --tag 1.6.0 --threshold 0.7

By default the script re-ingests the currently pinned
``corpus.SCIKIT_LEARN_TAG`` into the inactive color — the "rebuild"
form of blue/green that the Module 24 exercise uses to teach the cutover
mechanism. Pass ``--tag <version>`` to point the migration at a
different scikit-learn release, which is the actual version-upgrade
scenario Module 24 frames in the demo.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the project root importable when run directly (e.g.
# `uv run python scripts/migrate_blue_green.py`). The make targets get this from
# the Makefile's `export PYTHONPATH := .`; a direct script invocation does not.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.migrate import (
    DEFAULT_EVAL_SAMPLE_SIZE,
    DEFAULT_RECALL_THRESHOLD,
    migrate_blue_green,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Blue/green re-ingest of the ScikitDocs corpus.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="scikit-learn git tag to ingest (default: corpus.SCIKIT_LEARN_TAG)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_RECALL_THRESHOLD,
        help=f"recall@5 floor (default: {DEFAULT_RECALL_THRESHOLD})",
    )
    parser.add_argument(
        "--eval-sample-size",
        type=int,
        default=DEFAULT_EVAL_SAMPLE_SIZE,
        help=f"golden-set rows to evaluate (default: {DEFAULT_EVAL_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help="Do not drop the freshly-built color on gate failure (forensics)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(name)s: %(message)s",
    )
    outcome = migrate_blue_green(
        source_tag=args.tag,
        threshold=args.threshold,
        eval_sample_size=args.eval_sample_size,
        drop_failed=not args.keep_failed,
    )
    summary = (
        f"\n[migrate] target={outcome.target_color}\n"
        f"[migrate] recall@5={outcome.recall_at_k:.3f} "
        f"vs threshold={outcome.threshold:.3f} on n={outcome.eval_n}\n"
        f"[migrate] previous_active={outcome.previous_color}\n"
        f"[migrate] swapped={outcome.swapped}\n"
        f"[migrate] duration={outcome.duration_seconds:.1f}s\n"
    )
    print(summary)
    if not outcome.swapped:
        print(f"[migrate] reason: {outcome.reason}")
        return 1
    print(f"[migrate] alias now points at {outcome.target_color}")
    print(f"[migrate] state file: {Path('data/ACTIVE_COLLECTION').resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
