"""Wave-4 smoke RAG gate (REQ-063).

Embeds three representative queries from ``data/golden_set.csv``,
queries the Chroma collection directly (``src/pipeline.py`` is still
stubbed at this point in the wave), and asserts recall@5 ≥ 0.7. Writes
the per-query and aggregate result to ``data/SMOKE_REPORT.md``.

This script is the gate before downstream impl REQs (M03 / M05 / M07 /
M09 / M11 / M13 / M15 / M18 / M20 / M22 / M24 / M26) start touching
``src/``. The pinned tag ``wave-4-infra-frozen`` follows immediately
after.

Hit-any recall semantics: a query is "recalled" if at least
``min_hits`` of its ``expected_doc_ids`` candidates appear in the
top-k results. Doc_ids in the candidate set match by **prefix**
because the actual section anchors aren't fully predictable without
parsing every RST file — a near-prefix match is good enough for a
smoke gate.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

_saved_fd2 = os.dup(2)
_devnull = os.open(os.devnull, os.O_WRONLY)
try:
    os.dup2(_devnull, 2)
    import chromadb
    from chromadb.config import Settings as ChromaSettings
finally:
    sys.stderr.flush()
    os.dup2(_saved_fd2, 2)
    os.close(_devnull)
    os.close(_saved_fd2)

import logging

from openai import OpenAI

from scripts.load_data import COLLECTION_NAME, _make_openai_client  # type: ignore
from src.config import settings

logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

GOLDEN_SET_PATH: Path = Path("data/golden_set.csv")
SMOKE_REPORT_PATH: Path = Path("data/SMOKE_REPORT.md")
CHROMA_PATH: Path = Path(settings.chroma_path)

# Three representative queries — one factual, one procedural, one conceptual.
SMOKE_QUERIES: tuple[str, ...] = (
    "What is the default value of `n_estimators` in `RandomForestClassifier`?",
    "How do I scale features before clustering them?",
    "Why use stratified K-fold instead of plain K-fold for classification?",
)

TOP_K: int = 5
RECALL_FLOOR: float = 0.7


def load_golden_rows(path: Path) -> dict[str, dict]:
    by_question: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            by_question[row["question"]] = row
    return by_question


def hit_any(
    retrieved_ids: list[str], expected_ids: list[str], min_hits: int
) -> tuple[bool, list[str]]:
    """Return (passed, matched_pairs) under prefix-match hit-any semantics."""
    if not expected_ids:
        return (True, [])
    matches: list[str] = []
    for expected in expected_ids:
        expected = expected.strip()
        if not expected:
            continue
        for got in retrieved_ids:
            # Prefix match: golden expected_doc_id is a prefix of the
            # actual doc_id we got back. This is intentionally loose
            # because RST section anchors aren't easy to predict offline.
            if got == expected or got.startswith(expected + "."):
                matches.append(f"{got} ⇐ {expected}")
                break
    return (len(matches) >= min_hits, matches)


def embed_query(client: OpenAI, text: str) -> list[float]:
    response = client.embeddings.create(input=[text], model=settings.embedding_model)
    return response.data[0].embedding


def run_smoke_gate() -> tuple[bool, list[dict]]:
    golden = load_golden_rows(GOLDEN_SET_PATH)
    missing = [q for q in SMOKE_QUERIES if q not in golden]
    if missing:
        raise SystemExit(
            f"smoke queries not present in {GOLDEN_SET_PATH}: {missing}"
        )

    client_db = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = client_db.get_collection(name=COLLECTION_NAME)
    print(f"[smoke_gate] collection '{COLLECTION_NAME}' has {collection.count()} chunks")

    openai_client = _make_openai_client()
    results: list[dict] = []
    for question in SMOKE_QUERIES:
        row = golden[question]
        expected = [s for s in row["expected_doc_ids"].split("|") if s.strip()]
        min_hits = int(row["min_hits"])
        t0 = time.monotonic()
        q_vec = embed_query(openai_client, question)
        retrieved = collection.query(
            query_embeddings=[q_vec],
            n_results=TOP_K,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        retrieved_ids = retrieved["ids"][0]
        passed, matches = hit_any(retrieved_ids, expected, min_hits)
        results.append({
            "question": question,
            "query_type": row["query_type"],
            "expected_doc_ids": expected,
            "min_hits": min_hits,
            "retrieved_ids": retrieved_ids,
            "matches": matches,
            "passed": passed,
            "latency_ms": round(elapsed_ms, 1),
        })

    passes = sum(1 for r in results if r["passed"])
    recall = passes / len(results) if results else 0.0
    overall_passed = recall >= RECALL_FLOOR
    return overall_passed, results, recall


def write_report(passed: bool, results: list[dict], recall: float) -> None:
    timestamp = datetime.now(tz=UTC).isoformat(timespec="seconds")
    seeded_count = _count_seeded()
    lines: list[str] = []
    lines.append("# Wave 4 smoke RAG gate — SMOKE_REPORT")
    lines.append("")
    lines.append(f"- Run at: {timestamp}")
    lines.append(f"- Collection: `{COLLECTION_NAME}`")
    lines.append(f"- Embedding model: `{settings.embedding_model}`")
    lines.append(f"- Top-k: {TOP_K}")
    lines.append(f"- Recall floor: {RECALL_FLOOR:.2f}")
    lines.append(f"- Seeded chunks present: {seeded_count}")
    lines.append("")
    lines.append("## Overall result")
    lines.append("")
    status_word = "PASS" if passed else "FAIL"
    lines.append(f"**{status_word}** — recall@{TOP_K} = {recall:.2f} "
                 f"(threshold ≥ {RECALL_FLOOR:.2f})")
    lines.append("")
    lines.append("## Per-query detail")
    lines.append("")
    for idx, r in enumerate(results, start=1):
        lines.append(f"### Query {idx} — {r['query_type']}")
        lines.append("")
        lines.append(f"> {r['question']}")
        lines.append("")
        lines.append(f"- **Passed:** {'✅' if r['passed'] else '❌'} "
                     f"(min_hits={r['min_hits']})")
        lines.append(f"- **Latency:** {r['latency_ms']} ms")
        lines.append(f"- **Expected doc_id prefixes** ({len(r['expected_doc_ids'])}):")
        for e in r["expected_doc_ids"]:
            lines.append(f"  - `{e}`")
        lines.append(f"- **Retrieved top-{TOP_K} doc_ids:**")
        for got in r["retrieved_ids"]:
            lines.append(f"  - `{got}`")
        if r["matches"]:
            lines.append("- **Matches:**")
            for m in r["matches"]:
                lines.append(f"  - {m}")
        else:
            lines.append("- **Matches:** *(none)*")
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Hit-any semantics: a query passes if at least `min_hits` "
                 "of its expected doc_id prefixes match retrieved doc_ids.")
    lines.append("- Prefix matching is intentionally loose because real "
                 "section anchors are slugified from RST titles and not "
                 "fully predictable offline. The smoke gate validates the "
                 "*shape* of retrieval, not exact anchor alignment.")
    lines.append("- The 8 deliberately-seeded chunks (see "
                 "`SEEDING_NOTES.md`) are present in the collection so the "
                 "M11 top-k sweep has signal but the smoke floor is "
                 "preserved.")
    SMOKE_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[smoke_gate] wrote {SMOKE_REPORT_PATH}")


def _count_seeded() -> int:
    try:
        client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        collection = client.get_collection(name=COLLECTION_NAME)
        seeded = collection.get(where={"is_seeded": True}, include=[])
        return len(seeded["ids"])
    except Exception:
        return -1


def main() -> None:
    passed, results, recall = run_smoke_gate()
    write_report(passed, results, recall)
    print(f"[smoke_gate] recall@{TOP_K} = {recall:.2f} "
          f"(floor {RECALL_FLOOR:.2f})  —  "
          f"{'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
