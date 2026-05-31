"""Tests for the A/B routing primitives (REQ-073, M22).

The contract this REQ delivers:

- ``pick_variant`` is sticky for non-empty ``client_id`` — same input
  → same output, every call, forever.
- The salt parameter isolates concurrent experiments.
- ``client_id=None`` (or empty) falls back to weighted random sampling.
- Weight distribution is honoured within statistical tolerance on a
  10k-sample sweep.
- ``log_assignment`` writes a JSON-parseable row with the full schema
  the analyzer reads.

The tests are deterministic where they assert exact behavior (sticky
assertions, schema checks) and tolerance-banded where they assert
distributional behavior (split %, salt isolation).
"""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.models import Source, TokenUsage
from src.optimization import call_with_variant, log_assignment, pick_variant


# --- pick_variant: stickiness -----------------------------------------

def test_pick_variant_is_sticky_for_same_client_id() -> None:
    """Same ``client_id`` → same variant across many calls."""
    split = {"A": 0.5, "B": 0.5}
    first = pick_variant("user-42", split)
    for _ in range(100):
        assert pick_variant("user-42", split) == first


def test_pick_variant_is_deterministic_across_processes() -> None:
    """The hash is content-addressable — no process-local state."""
    # The actual cross-process check is impractical inside the unit
    # test, but the next-best thing is asserting that the function has
    # no random.* dependency when client_id is provided. We seed the
    # global RNG to a known state and assert the assignment is
    # unaffected by it.
    random.seed(0)
    first = pick_variant("user-123", {"A": 0.5, "B": 0.5})
    random.seed(99999)
    second = pick_variant("user-123", {"A": 0.5, "B": 0.5})
    assert first == second


# --- pick_variant: distribution + weights -----------------------------

def test_pick_variant_distribution_matches_traffic_split() -> None:
    """Over 10k hashed client_ids the empirical split is within ±3pp."""
    split = {"A": 0.7, "B": 0.3}
    counts = {"A": 0, "B": 0}
    for i in range(10_000):
        v = pick_variant(f"user-{i}", split)
        counts[v] += 1
    a_share = counts["A"] / 10_000
    assert 0.67 <= a_share <= 0.73, (
        f"Expected ~70% A, got {a_share:.3f} ({counts})"
    )


def test_pick_variant_normalizes_weights_that_dont_sum_to_one() -> None:
    """{'A': 10, 'B': 30} ≡ {'A': 0.25, 'B': 0.75}."""
    split_raw = {"A": 10, "B": 30}
    split_normalised = {"A": 0.25, "B": 0.75}
    counts_raw = {"A": 0, "B": 0}
    counts_norm = {"A": 0, "B": 0}
    for i in range(2_000):
        counts_raw[pick_variant(f"u-{i}", split_raw)] += 1
        counts_norm[pick_variant(f"u-{i}", split_normalised)] += 1
    # Identical assignments because the hash and the normalisation are
    # deterministic given the weight ratios — not just statistically
    # close, exactly equal.
    assert counts_raw == counts_norm


# --- pick_variant: salt isolation -------------------------------------

def test_pick_variant_salt_isolates_experiments() -> None:
    """Same client + different salt → independent variant choices."""
    split = {"A": 0.5, "B": 0.5}
    # Over 500 client_ids we expect roughly half to differ between two
    # salts. A binomial concentration argument says the count should
    # land in [200, 300] with overwhelming probability.
    differ = 0
    for i in range(500):
        cid = f"user-{i}"
        v1 = pick_variant(cid, split, salt="experiment-prompt-style")
        v2 = pick_variant(cid, split, salt="experiment-tier-mix")
        if v1 != v2:
            differ += 1
    assert 200 <= differ <= 300, (
        f"Salt did not appear to isolate assignments: {differ}/500 differed"
    )


def test_pick_variant_same_salt_same_assignment() -> None:
    """Same salt + same client → same variant."""
    split = {"A": 0.5, "B": 0.5}
    a = pick_variant("user-77", split, salt="exp-1")
    b = pick_variant("user-77", split, salt="exp-1")
    assert a == b


# --- pick_variant: fallback -------------------------------------------

def test_pick_variant_falls_back_to_per_request_when_client_id_is_none() -> None:
    """No client_id → weighted random sampling, distribution matches."""
    split = {"A": 0.5, "B": 0.5}
    random.seed(42)  # reproducible
    counts = {"A": 0, "B": 0}
    for _ in range(10_000):
        v = pick_variant(None, split)
        counts[v] += 1
    a_share = counts["A"] / 10_000
    assert 0.47 <= a_share <= 0.53, (
        f"Per-request fallback distribution off: {a_share:.3f}"
    )


def test_pick_variant_treats_empty_string_as_fallback() -> None:
    """Empty client_id behaves the same as None."""
    split = {"A": 1.0, "B": 0.0}
    # All weight on A — even the random path must hit A every time.
    assert pick_variant("", split) == "A"


# --- pick_variant: validation -----------------------------------------

def test_pick_variant_rejects_empty_traffic_split() -> None:
    with pytest.raises(ValueError):
        pick_variant("user-1", {})


def test_pick_variant_rejects_negative_weights() -> None:
    with pytest.raises(ValueError):
        pick_variant("user-1", {"A": 0.5, "B": -0.1})


def test_pick_variant_rejects_zero_total_weight() -> None:
    with pytest.raises(ValueError):
        pick_variant("user-1", {"A": 0.0, "B": 0.0})


# --- log_assignment ---------------------------------------------------

def _usage() -> TokenUsage:
    return TokenUsage(prompt_tokens=120, completion_tokens=45)


def test_log_assignment_writes_one_jsonl_row_per_call() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ab.jsonl"
        for i in range(3):
            log_assignment(
                path,
                client_id=f"user-{i}",
                variant="A",
                question=f"q{i}",
                answer=f"a{i}",
                usage=_usage(),
                cost_usd=0.0001,
                latency_ms=400 + i,
                success=True,
            )
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        rows = [json.loads(line) for line in lines]
        assert rows[1]["client_id"] == "user-1"
        assert rows[2]["latency_ms"] == 402


def test_log_assignment_creates_parent_dir() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nested" / "deep" / "ab.jsonl"
        log_assignment(
            path,
            client_id="user-1",
            variant="B",
            question="q",
            answer="a",
            usage=_usage(),
            cost_usd=0.0,
            latency_ms=100,
            success=False,
        )
        assert path.exists()


def test_log_assignment_schema_contains_required_fields() -> None:
    required = {
        "client_id",
        "variant",
        "question",
        "answer",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "cost_usd",
        "success",
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ab.jsonl"
        log_assignment(
            path,
            client_id=None,
            variant="A",
            question="q",
            answer="a",
            usage=_usage(),
            cost_usd=0.0001,
            latency_ms=200,
            success=True,
        )
        row = json.loads(path.read_text(encoding="utf-8").strip())
        assert set(row.keys()) == required
        # Null client_id is preserved (sticky-vs-fallback distinguisher).
        assert row["client_id"] is None


# --- call_with_variant ------------------------------------------------

def _fake_openai_response(text: str, p_tok: int = 120, c_tok: int = 45) -> Any:
    class FakeMessage:
        content = text

    class FakeChoice:
        message = FakeMessage()

    class FakeUsage:
        prompt_tokens = p_tok
        completion_tokens = c_tok

    class FakeResponse:
        choices = [FakeChoice()]
        usage = FakeUsage()

    return FakeResponse()


def test_call_with_variant_renders_correct_template() -> None:
    """Variant A and Variant B render different system prompts."""
    captured: dict[str, list[dict]] = {"A": [], "B": []}

    def fake_create(self, **kwargs):
        # Which variant we're testing is known from the system prompt
        # content — variant B has "Be expansive" instead of "Be concise".
        sys_msg = kwargs["messages"][0]["content"]
        if "Be expansive" in sys_msg:
            captured["B"].append(kwargs)
        else:
            captured["A"].append(kwargs)
        return _fake_openai_response("ok")

    sources = [Source(doc_id="d1", chunk_text="text", similarity_score=0.9)]
    with patch(
        "openai.resources.chat.completions.Completions.create",
        new=fake_create,
    ):
        call_with_variant("q?", sources, "A", model="gpt-4o-mini")
        call_with_variant("q?", sources, "B", model="gpt-4o-mini")

    assert len(captured["A"]) == 1
    assert len(captured["B"]) == 1
    assert "Be expansive" not in captured["A"][0]["messages"][0]["content"]
    assert "Be expansive" in captured["B"][0]["messages"][0]["content"]


def test_call_with_variant_returns_latency_ms() -> None:
    """The return tuple's fourth element is a non-negative int latency."""

    def fake_create(self, **kwargs):
        return _fake_openai_response("hello")

    sources = [Source(doc_id="d1", chunk_text="text", similarity_score=0.9)]
    with patch(
        "openai.resources.chat.completions.Completions.create",
        new=fake_create,
    ):
        answer, usage, cost, latency_ms = call_with_variant(
            "q?", sources, "A", model="gpt-4o-mini"
        )
    assert answer == "hello"
    assert usage.prompt_tokens == 120
    assert cost > 0
    assert isinstance(latency_ms, int)
    assert latency_ms >= 0
