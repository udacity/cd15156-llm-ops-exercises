"""A/B log analyzer for Module 22 / REQ-073.

Reads ``data/ab_log.jsonl`` (written by ``src.optimization.ab.log_assignment``
in the learner's Exercise 1 harness), builds a 2×2 variant-by-success
contingency table, runs ``scipy.stats.chi2_contingency``, and prints
per-variant aggregates (n, mean and median latency, mean and total cost,
mean completion tokens).

Run via ``make ab-analyze`` (Makefile target) or directly:

    uv run python scripts/ab_analyze.py
    uv run python scripts/ab_analyze.py --log data/custom_log.jsonl

The script is read-only and idempotent — running it twice produces
identical output as long as the log file hasn't changed.
"""

import argparse
import json
from pathlib import Path
from statistics import mean, median

from scipy.stats import chi2_contingency


def load_rows(log_path: Path) -> list[dict]:
    """Parse one JSON object per non-empty line."""
    if not log_path.exists():
        raise SystemExit(
            f"Log file not found: {log_path}. "
            "Run scripts/ab_simulate.py (Exercise 1) first."
        )
    rows: list[dict] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    if not rows:
        raise SystemExit(
            f"Log file is empty: {log_path}. "
            "Did the simulation actually run?"
        )
    return rows


def contingency_table(
    rows: list[dict], variants: list[str]
) -> list[list[int]]:
    """Build a variants × [success, fail] table."""

    def cell(variant: str, success: bool) -> int:
        return sum(
            1
            for r in rows
            if r["variant"] == variant and bool(r["success"]) is success
        )

    return [[cell(v, True), cell(v, False)] for v in variants]


def per_variant_stats(rows: list[dict], variant: str) -> dict:
    """Return n, latency, cost, and token aggregates for one variant."""
    these = [r for r in rows if r["variant"] == variant]
    if not these:
        return {"n": 0}
    return {
        "n": len(these),
        "mean_latency_ms": int(mean(r["latency_ms"] for r in these)),
        "p50_latency_ms": int(median(r["latency_ms"] for r in these)),
        "mean_cost_usd": mean(r["cost_usd"] for r in these),
        "total_cost_usd": sum(r["cost_usd"] for r in these),
        "mean_completion_tokens": int(
            mean(r["completion_tokens"] for r in these)
        ),
    }


def unique_client_count(rows: list[dict]) -> int:
    """Count distinct non-null ``client_id`` values.

    For sticky-by-user A/B the *effective* sample size is the unique-
    client count, not the raw call count. Surfacing it lets the
    learner reason about whether the test is underpowered for the
    sticky tradeoff Module 21 V3 named.
    """
    return len({r.get("client_id") for r in rows if r.get("client_id")})


def print_report(rows: list[dict]) -> None:
    variants = sorted({r["variant"] for r in rows})
    if len(variants) < 2:
        raise SystemExit(
            f"Need at least 2 variants in the log, found: {variants}"
        )

    print(f"Loaded {len(rows)} rows from the A/B log.")
    print(f"Unique client_ids (sticky effective N): {unique_client_count(rows)}")
    print()

    table = contingency_table(rows, variants)
    result = chi2_contingency(table)
    for variant, (succ, fail) in zip(variants, table):
        total = succ + fail
        rate = succ / total if total else 0.0
        print(f"Variant {variant}: {succ}/{total} success ({rate:.1%})")
    print(f"chi2 statistic:                {result.statistic:.3f}")
    print(f"p-value:                       {result.pvalue:.4f}")
    print(f"degrees of freedom:            {result.dof}")
    print(f"significant at alpha=0.05:     {result.pvalue < 0.05}")
    print()

    header = f"{'metric':<24}" + "".join(f"{v:>12}" for v in variants)
    print(header)
    print("-" * len(header))
    metrics = [
        "n",
        "mean_latency_ms",
        "p50_latency_ms",
        "mean_cost_usd",
        "total_cost_usd",
        "mean_completion_tokens",
    ]
    stats = {v: per_variant_stats(rows, v) for v in variants}
    for m in metrics:
        row = f"{m:<24}"
        for v in variants:
            val = stats[v].get(m, "—")
            if isinstance(val, float):
                row += f"{val:>12.5f}"
            else:
                row += f"{val:>12}"
        print(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("data/ab_log.jsonl"),
        help="Path to the JSONL log written by the harness.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.log)
    print_report(rows)


if __name__ == "__main__":
    main()
