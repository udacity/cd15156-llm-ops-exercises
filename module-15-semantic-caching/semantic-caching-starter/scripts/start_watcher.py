"""CLI entry for the docs-inbox watcher (Module 24).

Usage:
    uv run python scripts/start_watcher.py
    uv run python scripts/start_watcher.py --inbox data/docs_inbox --log-level DEBUG

The watcher blocks on the main thread until SIGINT (Ctrl+C) or
SIGTERM. Both signals set the same stop event so a half-processed
event finishes cleanly before the observer thread joins.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path

from src.ingestion import (
    ACTIVE_COLLECTION_PATH,
    read_active_collection,
    start_observer,
)
from src.ingestion.watcher import ingest_existing

DEFAULT_INBOX = Path("data/docs_inbox")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Watch a docs inbox and ingest dropped JSON sections."
    )
    parser.add_argument(
        "--inbox",
        type=Path,
        default=DEFAULT_INBOX,
        help=f"Inbox directory to watch (default: {DEFAULT_INBOX})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--no-initial-sweep",
        action="store_true",
        help="Skip ingesting JSONs that already exist in the inbox at startup",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(name)s: %(message)s",
    )

    active = read_active_collection()
    alias_path = ACTIVE_COLLECTION_PATH
    print(
        f"[watcher] active collection: {active} "
        f"({'alias unset → legacy scikit_docs' if not alias_path.exists() else f'via {alias_path}'})"
    )
    print(f"[watcher] watching {args.inbox} for new section JSONs")

    if not args.no_initial_sweep:
        count = sum(1 for _ in ingest_existing(args.inbox))
        if count:
            print(f"[watcher] initial sweep processed {count} pre-existing file(s)")

    stop_event = threading.Event()
    observer = start_observer(args.inbox, stop_event=stop_event)

    def _handle_signal(signum: int, _frame: object) -> None:
        print(f"\n[watcher] received signal {signum}; stopping observer ...")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    finally:
        observer.stop()
        observer.join(timeout=5)
        print("[watcher] stopped cleanly")


if __name__ == "__main__":
    sys.exit(main())
