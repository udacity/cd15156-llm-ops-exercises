"""Unit tests for ``src.pricing`` + ``src.cost.{tracker,dashboard}`` (Module 13).

Covers the pure-function surface: cost math, JSONL round-trip,
summarization, HTML rendering on empty and populated inputs. The
``/cost-dashboard`` route itself is integration-tested at Module 18
when the gateway lands.
"""

import json
from pathlib import Path

import pytest

from src.cost.dashboard import render_html
from src.cost.tracker import load_log, log_request, summarize
from src.models import TokenUsage
from src.pricing import MODEL_PRICING, compute_cost


# === src/pricing.py ===


def test_compute_cost_known_models() -> None:
    """gpt-4o costs (2.50 * 1000 + 10.00 * 500) / 1_000_000 = 0.0075."""
    usage = TokenUsage(prompt_tokens=1000, completion_tokens=500)
    assert compute_cost("gpt-4o", usage) == pytest.approx(0.0075)
    # gpt-4o-mini is ~17× cheaper on input, ~17× cheaper on output.
    assert compute_cost("gpt-4o-mini", usage) == pytest.approx(0.00045)


def test_compute_cost_unknown_model_raises_keyerror() -> None:
    """Deliberate: a typo in .env should fail loudly, not silently log $0."""
    usage = TokenUsage(prompt_tokens=100, completion_tokens=10)
    with pytest.raises(KeyError):
        compute_cost("gpt-99-imaginary", usage)


def test_model_pricing_table_shape() -> None:
    """Every entry is ``(input_per_million, output_per_million)`` with output ≥ input."""
    for model, (input_rate, output_rate) in MODEL_PRICING.items():
        assert input_rate > 0, f"{model} input rate not positive"
        assert output_rate >= input_rate, f"{model} output ({output_rate}) < input ({input_rate})"


# === src/cost/tracker.py ===


def test_log_request_writes_one_jsonl_row(tmp_path: Path) -> None:
    log = tmp_path / "cost_log.jsonl"
    usage = TokenUsage(prompt_tokens=120, completion_tokens=40)
    record = log_request("gpt-4o-mini", usage, 0.000042, "simple", path=log)

    assert log.exists()
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    written = json.loads(lines[0])
    assert written["model"] == "gpt-4o-mini"
    assert written["prompt_tokens"] == 120
    assert written["completion_tokens"] == 40
    assert written["cost_usd"] == pytest.approx(0.000042)
    assert written["query_type"] == "simple"
    assert "timestamp" in written
    assert record == written


def test_log_request_appends(tmp_path: Path) -> None:
    log = tmp_path / "cost_log.jsonl"
    usage = TokenUsage(prompt_tokens=100, completion_tokens=20)
    for _ in range(3):
        log_request("gpt-4o", usage, 0.0005, "complex", path=log)
    assert len(log.read_text(encoding="utf-8").strip().splitlines()) == 3


def test_log_request_creates_parent_dir(tmp_path: Path) -> None:
    log = tmp_path / "nested" / "deep" / "cost_log.jsonl"
    log_request("gpt-4o-mini", TokenUsage(prompt_tokens=50, completion_tokens=10),
                0.00003, "simple", path=log)
    assert log.exists()


def test_load_log_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_log(tmp_path / "does_not_exist.jsonl") == []


def test_load_log_round_trips(tmp_path: Path) -> None:
    log = tmp_path / "cost_log.jsonl"
    written = []
    for i in range(5):
        usage = TokenUsage(prompt_tokens=100 + i, completion_tokens=20 + i)
        written.append(log_request("gpt-4o-mini", usage, 0.0001 * (i + 1),
                                   "simple", path=log))
    loaded = load_log(log)
    assert loaded == written


# === summarize ===


def test_summarize_empty_log() -> None:
    summary = summarize([])
    assert summary["total_requests"] == 0
    assert summary["total_cost_usd"] == 0.0
    assert summary["by_model"] == {}


def test_summarize_groups_by_model() -> None:
    records = [
        {"model": "gpt-4o", "cost_usd": 0.01, "prompt_tokens": 100, "completion_tokens": 50},
        {"model": "gpt-4o", "cost_usd": 0.02, "prompt_tokens": 200, "completion_tokens": 100},
        {"model": "gpt-4o-mini", "cost_usd": 0.001, "prompt_tokens": 100, "completion_tokens": 50},
    ]
    summary = summarize(records)
    assert summary["total_requests"] == 3
    assert summary["total_cost_usd"] == pytest.approx(0.031)
    assert summary["by_model"]["gpt-4o"]["requests"] == 2
    assert summary["by_model"]["gpt-4o"]["cost_usd"] == pytest.approx(0.03)
    assert summary["by_model"]["gpt-4o"]["avg_cost_usd"] == pytest.approx(0.015)
    assert summary["by_model"]["gpt-4o-mini"]["requests"] == 1


# === src/cost/dashboard.py ===


def test_render_html_empty_summary() -> None:
    html = render_html({"total_requests": 0, "total_cost_usd": 0.0, "by_model": {}})
    assert "<!doctype html>" in html
    assert "Cost Dashboard" in html
    assert "No requests logged yet" in html


def test_render_html_populated_summary() -> None:
    summary = {
        "total_requests": 2,
        "total_cost_usd": 0.0042,
        "by_model": {
            "gpt-4o": {"requests": 1, "cost_usd": 0.004, "avg_cost_usd": 0.004},
            "gpt-4o-mini": {"requests": 1, "cost_usd": 0.0002, "avg_cost_usd": 0.0002},
        },
    }
    html = render_html(summary)
    assert "$0.0042" in html
    assert "gpt-4o" in html
    assert "gpt-4o-mini" in html
    # Per-model rows must include the formatted costs.
    assert "$0.0040" in html
    assert "$0.0002" in html


def test_render_html_escapes_model_names() -> None:
    """Defense-in-depth: model names from the log get HTML-escaped."""
    summary = {
        "total_requests": 1,
        "total_cost_usd": 0.001,
        "by_model": {
            "<script>alert(1)</script>": {"requests": 1, "cost_usd": 0.001, "avg_cost_usd": 0.001},
        },
    }
    html = render_html(summary)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
