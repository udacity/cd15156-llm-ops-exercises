"""Filesystem watcher for ad-hoc section drops (REQ-074, M24).

The watcher mirrors Module 23's producer–queue–consumer pattern at the
inbox layer. A JSON file dropped into ``data/docs_inbox/`` represents
one pre-chunked scikit-learn doc section — the schema matches what
:func:`src.corpus.load_corpus` yields, so the watcher path is the
single-file analog of the bulk ``make load-data`` path. Successful
ingestions land in whichever color the alias currently points at; bad
inputs are quarantined under ``data/docs_inbox/failed/`` with a sibling
``.error.txt`` recording the reason.

Idempotency rides on **content-hashed ids**. The chunk id is
``"{doc_id}#{sha256(text)[:12]}"``; a re-drop of the same JSON produces
the same id and Chroma's upsert overwrites the prior row instead of
duplicating it. Drift-fixing the same section means a new hash, new id,
and the previous row stays in the index until the next migration drops
the inactive color — the deliberately conservative choice for a live
serving collection. M24 names this trade-off in the exercise.

The watcher reuses ``scripts/load_data.embed_missing`` for the embedding
call so the disk cache at ``data/embedding_cache.jsonl`` is shared with
the bulk path; one ingest in the watcher costs <50ms of OpenAI time if
the section has been seen before.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from src import store

logger = logging.getLogger("ingestion.watcher")

QUARANTINE_SUBDIR: str = "failed"
ERROR_SUFFIX: str = ".error.txt"
INBOX_GLOB: str = "*.json"

# Schema contract — matches one ``corpus.load_corpus`` yield. Drops
# missing any of these fields land in ``failed/`` with a reason.
REQUIRED_FIELDS: tuple[str, ...] = ("doc_id", "text", "metadata")
REQUIRED_METADATA_FIELDS: tuple[str, ...] = (
    "source_path",
    "section_title",
    "url",
)

# A drop larger than this is almost certainly a producer-side mistake
# (a whole-corpus dump pointed at the per-file inbox). The 256 KB ceiling
# is two orders of magnitude above the largest single section in the
# pinned scikit-learn 1.5.2 corpus.
MAX_FILE_BYTES: int = 256 * 1024

# Settle-window for in-flight writes — ``cp large.json inbox/`` may fire
# the create event before the bytes are all flushed.
SETTLE_POLL_SECONDS: float = 0.1
SETTLE_TIMEOUT_SECONDS: float = 5.0


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of :func:`validate_section`. Either valid or has a reason."""

    valid: bool
    reason: str = ""

    @classmethod
    def ok(cls) -> "ValidationResult":
        return cls(valid=True)

    @classmethod
    def fail(cls, reason: str) -> "ValidationResult":
        return cls(valid=False, reason=reason)


def validate_section(payload: object) -> ValidationResult:
    """Two-stage check: top-level shape, then metadata sub-shape.

    The first failure short-circuits — there's no point checking the
    metadata of a payload that isn't even a dict. The returned reason
    string is what gets written to the quarantine's ``.error.txt``
    sibling and surfaced in the watcher log.
    """
    if not isinstance(payload, dict):
        return ValidationResult.fail(
            f"top-level payload must be a JSON object; got {type(payload).__name__}"
        )
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        return ValidationResult.fail(f"missing required fields: {missing!r}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return ValidationResult.fail(
            "'metadata' must be a JSON object"
        )
    missing_meta = [f for f in REQUIRED_METADATA_FIELDS if f not in metadata]
    if missing_meta:
        return ValidationResult.fail(
            f"missing required metadata fields: {missing_meta!r}"
        )
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return ValidationResult.fail("'text' must be a non-empty string")
    return ValidationResult.ok()


def chunk_id_for(doc_id: str, text: str) -> str:
    """``"<doc_id>#<sha256-12>"`` — re-drop of identical text → same id."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{doc_id}#{digest}"


def _flatten_metadata(metadata: dict) -> dict:
    """Chroma stores scalar-only metadata; serialise list/dict values."""
    flat: dict = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            flat[key] = value
        else:
            flat[key] = json.dumps(value)
    return flat


def _wait_for_stable(path: Path) -> None:
    """Block until ``path``'s (size, mtime) repeats. Bounded by timeout."""
    import time

    deadline = time.monotonic() + SETTLE_TIMEOUT_SECONDS
    last: tuple[int, float] | None = None
    while time.monotonic() < deadline:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return
        signature = (stat.st_size, stat.st_mtime)
        if signature == last:
            return
        last = signature
        time.sleep(SETTLE_POLL_SECONDS)


def _quarantine(path: Path, inbox_dir: Path, reason: str) -> Path:
    """Move ``path`` into ``inbox_dir/failed/`` with a ``.error.txt`` sibling."""
    quarantine_dir = inbox_dir / QUARANTINE_SUBDIR
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    dest = quarantine_dir / path.name
    if dest.exists():
        dest.unlink()
    shutil.move(str(path), str(dest))
    error_path = dest.with_name(dest.name + ERROR_SUFFIX)
    error_path.write_text(reason + "\n", encoding="utf-8")
    return dest


def _embed_one(text: str) -> list[float]:
    """Embed a single chunk, sharing the bulk ingest's disk cache.

    Re-uses ``scripts/load_data.embed_missing`` so the cache file at
    ``data/embedding_cache.jsonl`` stays the one source of truth for
    seen-already chunks across the watcher path and the bulk path.
    """
    import importlib

    load_data = importlib.import_module("scripts.load_data")
    cache = load_data.load_embedding_cache(load_data.EMBEDDING_CACHE_PATH)
    client = load_data._make_openai_client()
    embeddings = load_data.embed_missing(client, [{"text": text}], cache)
    return embeddings[0]


def ingest_file(path: Path, inbox_dir: Path) -> bool:
    """Validate, embed, and upsert one inbox JSON. Quarantine on failure.

    Returns ``True`` on success, ``False`` on quarantine. Both cases
    log; callers don't need to inspect the return value unless they
    care for tests.
    """
    try:
        if not path.exists():
            return False
        if path.stat().st_size > MAX_FILE_BYTES:
            reason = f"file too large: {path.stat().st_size} bytes > {MAX_FILE_BYTES}"
            _quarantine(path, inbox_dir, reason)
            logger.warning("Quarantined %s: %s", path.name, reason)
            return False
        _wait_for_stable(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            reason = f"invalid JSON: {exc.msg} at line {exc.lineno} col {exc.colno}"
            _quarantine(path, inbox_dir, reason)
            logger.warning("Quarantined %s: %s", path.name, reason)
            return False
        check = validate_section(payload)
        if not check.valid:
            _quarantine(path, inbox_dir, check.reason)
            logger.warning("Quarantined %s: %s", path.name, check.reason)
            return False
        doc_id = payload["doc_id"]
        text = payload["text"]
        metadata = _flatten_metadata(payload["metadata"])
        chunk_id = chunk_id_for(doc_id, text)
        embedding = _embed_one(text)
        store.get_collection().upsert(
            ids=[chunk_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
        )
        logger.info(
            "Ingested %s → %s (active=%s)",
            path.name,
            chunk_id,
            store._resolve_alias(store.ALIAS_NAME),
        )
        return True
    except Exception as exc:  # noqa: BLE001
        reason = f"unhandled error: {type(exc).__name__}: {exc}"
        try:
            _quarantine(path, inbox_dir, reason)
        except OSError:
            pass
        logger.exception("Quarantined %s after unhandled error", path.name)
        return False


class InboxHandler(FileSystemEventHandler):
    """Dispatch ``.json`` creations to :func:`ingest_file`.

    ``recursive=False`` is set on the ``Observer`` in
    :func:`start_observer`; this handler does **not** inspect the
    ``failed/`` subdirectory itself, but the recursive-False setting is
    where that boundary actually lives.
    """

    def __init__(self, inbox_dir: Path) -> None:
        super().__init__()
        self._inbox_dir = inbox_dir

    def _handle(self, src_path: str) -> None:
        path = Path(src_path)
        if path.suffix != ".json":
            return
        if path.name.endswith(ERROR_SUFFIX):
            return
        ingest_file(path, self._inbox_dir)

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._handle(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        dest = getattr(event, "dest_path", None)
        if dest:
            self._handle(dest)


def start_observer(
    inbox_dir: Path,
    stop_event: threading.Event | None = None,
) -> Observer:
    """Start a watchdog observer on ``inbox_dir`` and return it.

    The caller owns the returned observer's lifecycle — either join the
    thread or let the signal handler in ``scripts/start_watcher.py``
    set ``stop_event`` and call ``observer.stop()``.
    """
    inbox_dir.mkdir(parents=True, exist_ok=True)
    handler = InboxHandler(inbox_dir)
    observer = Observer()
    observer.schedule(handler, str(inbox_dir), recursive=False)
    observer.start()
    return observer


def ingest_existing(inbox_dir: Path) -> Iterable[Path]:
    """Sweep ``inbox_dir`` for any JSONs that pre-date the watcher start.

    Yields the path of every file that was processed (whether ingested
    or quarantined). Callers can ignore the return value or count it.
    """
    inbox_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(inbox_dir.glob(INBOX_GLOB)):
        if path.is_file():
            ingest_file(path, inbox_dir)
            yield path
