"""Unit tests for the Module 24 ingestion surface.

These tests do not call out to OpenAI or stand up a real Chroma store
on disk for the embedding-bearing paths — the embedding call is patched
through ``monkeypatch`` because the only thing the alias / validation /
chunk-id surface needs to assert is the *control flow*, not the vector
quality. The migrate end-to-end happy path is covered by the runtime
gate (``make verify``) when the corpus is loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ingestion import alias as alias_mod
from src.ingestion import watcher as watcher_mod
from src.ingestion.alias import (
    BLUE_NAME,
    GREEN_NAME,
    other_color,
    read_active_collection,
    swap_alias,
)
from src.ingestion.watcher import (
    QUARANTINE_SUBDIR,
    chunk_id_for,
    validate_section,
)


# ---------- alias file management ---------------------------------------------


def test_read_active_collection_missing_file_falls_back(tmp_path: Path) -> None:
    missing = tmp_path / "ACTIVE_COLLECTION"
    assert read_active_collection(missing) == "scikit_docs"


def test_swap_alias_writes_atomically(tmp_path: Path) -> None:
    path = tmp_path / "ACTIVE_COLLECTION"
    previous = swap_alias(BLUE_NAME, path=path)
    assert previous == "scikit_docs"
    assert path.read_text().strip() == BLUE_NAME
    previous = swap_alias(GREEN_NAME, path=path)
    assert previous == BLUE_NAME
    assert path.read_text().strip() == GREEN_NAME
    # tmp sidecar should not linger after rename
    assert not list(tmp_path.glob("*.tmp"))


def test_swap_alias_rejects_invalid_target(tmp_path: Path) -> None:
    path = tmp_path / "ACTIVE_COLLECTION"
    with pytest.raises(ValueError):
        swap_alias("scikit_docs", path=path)
    with pytest.raises(ValueError):
        swap_alias("scikit_docs_red", path=path)
    assert not path.exists()


def test_other_color_round_trips() -> None:
    assert other_color(BLUE_NAME) == GREEN_NAME
    assert other_color(GREEN_NAME) == BLUE_NAME
    # Bootstrap: alias name or empty defaults to blue (first migration target)
    assert other_color("scikit_docs") == BLUE_NAME
    assert other_color("") == BLUE_NAME


# ---------- validation --------------------------------------------------------


def _good_payload() -> dict:
    return {
        "doc_id": "modules.x.intro",
        "text": "Some body text long enough to be plausibly a section.",
        "metadata": {
            "source_path": "doc/modules/x.rst",
            "section_title": "Intro",
            "url": "https://scikit-learn.org/stable/modules/x.html",
        },
    }


def test_validate_section_happy_path() -> None:
    result = validate_section(_good_payload())
    assert result.valid is True
    assert result.reason == ""


def test_validate_section_rejects_non_dict() -> None:
    result = validate_section(["doc_id"])
    assert result.valid is False
    assert "JSON object" in result.reason


def test_validate_section_rejects_missing_top_level_field() -> None:
    payload = _good_payload()
    del payload["metadata"]
    result = validate_section(payload)
    assert result.valid is False
    assert "metadata" in result.reason


def test_validate_section_rejects_missing_metadata_field() -> None:
    payload = _good_payload()
    del payload["metadata"]["url"]
    result = validate_section(payload)
    assert result.valid is False
    assert "url" in result.reason


def test_validate_section_rejects_empty_text() -> None:
    payload = _good_payload()
    payload["text"] = "   "
    result = validate_section(payload)
    assert result.valid is False
    assert "text" in result.reason


# ---------- chunk-id content hashing -----------------------------------------


def test_chunk_id_for_is_stable_for_same_text() -> None:
    a = chunk_id_for("modules.x.intro", "hello world")
    b = chunk_id_for("modules.x.intro", "hello world")
    assert a == b


def test_chunk_id_for_differs_when_text_differs() -> None:
    a = chunk_id_for("modules.x.intro", "hello world")
    b = chunk_id_for("modules.x.intro", "hello world!")
    assert a != b


# ---------- ingest_file: validation + quarantine paths -----------------------


def test_ingest_file_quarantines_invalid_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inbox = tmp_path / "docs_inbox"
    inbox.mkdir()
    bad = inbox / "broken.json"
    bad.write_text("{not valid json,}", encoding="utf-8")

    # Embedding + store path should never be reached.
    monkeypatch.setattr(
        watcher_mod, "_embed_one", lambda text: pytest.fail("should not embed")
    )

    assert watcher_mod.ingest_file(bad, inbox) is False
    assert not bad.exists()
    quarantined = inbox / QUARANTINE_SUBDIR / "broken.json"
    error = inbox / QUARANTINE_SUBDIR / "broken.json.error.txt"
    assert quarantined.exists()
    assert error.exists()
    assert "invalid JSON" in error.read_text()


def test_ingest_file_quarantines_schema_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inbox = tmp_path / "docs_inbox"
    inbox.mkdir()
    payload = _good_payload()
    del payload["metadata"]["url"]
    bad = inbox / "missing_url.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        watcher_mod, "_embed_one", lambda text: pytest.fail("should not embed")
    )
    assert watcher_mod.ingest_file(bad, inbox) is False
    error = inbox / QUARANTINE_SUBDIR / "missing_url.json.error.txt"
    assert "url" in error.read_text()


def test_ingest_file_quarantines_oversized_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inbox = tmp_path / "docs_inbox"
    inbox.mkdir()
    huge = inbox / "huge.json"
    huge.write_bytes(b"x" * (watcher_mod.MAX_FILE_BYTES + 1))

    monkeypatch.setattr(
        watcher_mod, "_embed_one", lambda text: pytest.fail("should not embed")
    )
    assert watcher_mod.ingest_file(huge, inbox) is False
    assert (inbox / QUARANTINE_SUBDIR / "huge.json").exists()


# ---------- ingest_file: happy path with mocked embed + store ----------------


class _FakeCollection:
    def __init__(self) -> None:
        self.upserts: list[dict] = []

    def upsert(self, **kwargs: object) -> None:
        self.upserts.append(kwargs)


def test_ingest_file_happy_path_upserts_with_content_hash_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inbox = tmp_path / "docs_inbox"
    inbox.mkdir()
    payload = _good_payload()
    good = inbox / "good.json"
    good.write_text(json.dumps(payload), encoding="utf-8")

    fake = _FakeCollection()
    monkeypatch.setattr(watcher_mod.store, "get_collection", lambda: fake)
    monkeypatch.setattr(watcher_mod, "_embed_one", lambda text: [0.1] * 1536)

    assert watcher_mod.ingest_file(good, inbox) is True
    assert len(fake.upserts) == 1
    ids = fake.upserts[0]["ids"]
    expected = chunk_id_for(payload["doc_id"], payload["text"])
    assert ids == [expected]
    # File stays in the inbox on success (audit trail).
    assert good.exists()
