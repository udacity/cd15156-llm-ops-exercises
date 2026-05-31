"""Operational ingestion surface for the ScikitDocs corpus (REQ-074, M24).

Two related concerns live here:

* :mod:`src.ingestion.alias` — read and atomically swap the
  ``data/ACTIVE_COLLECTION`` file that names which color
  (``scikit_docs_blue`` or ``scikit_docs_green``) the public
  ``scikit_docs`` alias resolves to.

* :mod:`src.ingestion.watcher` — a watchdog ``Observer`` that ingests
  pre-chunked JSON sections dropped into ``data/docs_inbox/`` and
  upserts them into the **active** color via content-hashed ids. The
  idempotent-receiver shape mirrors the producer–queue–consumer pattern
  named in Module 23's concept walk.

* :mod:`src.ingestion.migrate` — build the inactive color from a
  pinned ``scikit-learn`` source tag, run a small recall gate against
  the golden set, and atomically swap the alias on pass.
"""

from src.ingestion.alias import (
    ACTIVE_COLLECTION_PATH,
    BLUE_NAME,
    GREEN_NAME,
    other_color,
    read_active_collection,
    swap_alias,
)
from src.ingestion.migrate import (
    MigrationOutcome,
    migrate_blue_green,
    recall_at_k,
)
from src.ingestion.watcher import (
    QUARANTINE_SUBDIR,
    REQUIRED_FIELDS,
    InboxHandler,
    ingest_file,
    start_observer,
    validate_section,
)

__all__ = [
    "ACTIVE_COLLECTION_PATH",
    "BLUE_NAME",
    "GREEN_NAME",
    "InboxHandler",
    "MigrationOutcome",
    "QUARANTINE_SUBDIR",
    "REQUIRED_FIELDS",
    "ingest_file",
    "migrate_blue_green",
    "other_color",
    "read_active_collection",
    "recall_at_k",
    "start_observer",
    "swap_alias",
    "validate_section",
]
