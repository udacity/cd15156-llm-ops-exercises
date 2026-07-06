# TODO(m20-exercise-3): cost-amplification measurement — fire a baseline cohort and
# a long-generation cohort uncapped, then derive an output cap from the observed
# answer-length distribution (mean + N*sd) and re-fire to show it clip the tail.
# This is the reference solution for Exercise 3, Step 1; the starter ships a scaffold
# under the same marker.
"""Measure the LLM10 cost-amplification surface and set a data-driven output cap.

Fires cohorts through the traced pipeline with a fixed model, in three passes:

  1. baseline              — ordinary questions (normal answer shape)
  2. long, uncapped        — exhaustive-tutorial prompts with max_tokens=None
  3. long, capped          — the same prompts capped at mean + (--sd)*sd of pass 2

Pass 2 measures how long the answers actually run; pass 3 sets the cap from that
distribution so it only bites the tail, the way you would set an output cap in
production. It prints median and p95 cost per cohort, the amplification factor, and
which rows the cap clipped.

Calls traced_pipeline directly (the gateway's entry), so it needs no running server,
but it does need a live OPENAI_API_KEY and a loaded corpus (make load-data). The
uncapped long rows generate large answers on purpose, so expect a few cents.

Usage:
    PYTHONPATH=. uv run python scripts/cost_amplification.py
    PYTHONPATH=. uv run python scripts/cost_amplification.py --sd 0.5   # tighter cap, more clipping
"""

import argparse
import math
import statistics

from src.tracing import traced_pipeline

MODEL = "gpt-4o-mini"

BASELINE = [
    "What does StandardScaler.fit do?",
    "What is the default value of n_estimators in RandomForestClassifier?",
    "Which kernel does SVR use by default?",
    "What does fit_transform return?",
    "What is the default number of clusters for KMeans?",
    "What does train_test_split return?",
    "What is the default solver for LogisticRegression?",
    "What does OneHotEncoder do with unknown categories?",
    "What is the default criterion for DecisionTreeClassifier?",
    "What does cross_val_score return?",
]

_ESTIMATORS = [
    "RandomForestClassifier", "GradientBoostingClassifier", "SVC", "KMeans",
    "LogisticRegression", "DecisionTreeClassifier", "StandardScaler",
    "GridSearchCV", "Pipeline", "KNeighborsClassifier",
]
LONG = [
    f"Write a thorough tutorial covering every parameter of {est}, "
    "with worked code examples for each one."
    for est in _ESTIMATORS
]


def measure(questions: list[str], max_tokens: int | None, label: str) -> list[tuple[int, float]]:
    """Fire one cohort at a fixed cap, printing per-row (tokens, cost). Returns them."""
    cap = "uncapped" if max_tokens is None else f"cap={max_tokens}"
    print(f"\n{label} ({cap}, {len(questions)} rows)...", flush=True)
    rows = []
    for i, question in enumerate(questions, 1):
        resp = traced_pipeline(question, model=MODEL, max_tokens=max_tokens)
        tok = resp.tokens.completion_tokens
        rows.append((tok, resp.cost_usd))
        clipped = max_tokens is not None and tok >= max_tokens
        flag = "  <- clipped at cap" if clipped else ""
        print(f"  {i:>2}/{len(questions)}  {tok:>4} tok  ${resp.cost_usd:.4f}{flag}", flush=True)
    return rows


def p95(values: list[float]) -> float:
    s = sorted(values)
    k = math.ceil(0.95 * len(s)) - 1
    return s[max(0, min(k, len(s) - 1))]


def _row(label: str, costs: list[float]) -> str:
    return f"| {label:<16} | ${statistics.median(costs):.4f} | ${p95(costs):.4f} |"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10, help="rows per cohort (1-10)")
    parser.add_argument("--sd", type=float, default=1.0,
                        help="cap = mean + this many sd of the uncapped run (lower = more clipping)")
    args = parser.parse_args()
    n = max(1, min(args.n, 10))

    baseline = measure(BASELINE[:n], None, "baseline")
    uncapped = measure(LONG[:n], None, "long-generation")

    toks = [t for t, _ in uncapped]
    mean, sd = statistics.mean(toks), (statistics.stdev(toks) if len(toks) > 1 else 0.0)
    cap = round(mean + args.sd * sd)
    print(
        f"\nuncapped answer length: mean {mean:.0f} tokens, sd {sd:.0f}. "
        f"Setting the cap at mean + {args.sd:g} sd = {cap} tokens so it clips only the tail."
    )

    capped = measure(LONG[:n], cap, f"long-generation, data-driven cap")
    n_clipped = sum(1 for t, _ in capped if t >= cap)

    base_p95 = p95([c for _, c in baseline])
    print("\n=== cost per request ===")
    print("| cohort           | median  | p95     |")
    print("| ---------------- | ------- | ------- |")
    print(_row("baseline", [c for _, c in baseline]))
    print(_row("long, uncapped", [c for _, c in uncapped]))
    print(_row(f"long, cap {cap}", [c for _, c in capped]))

    if base_p95 > 0:
        print(
            f"\nAmplification (long p95 / baseline p95):  "
            f"uncapped {p95([c for _, c in uncapped]) / base_p95:.1f}x  ->  "
            f"capped {p95([c for _, c in capped]) / base_p95:.1f}x"
        )
    print(
        f"The cap at {cap} tokens clipped {n_clipped} of {n} long answers. Set it from "
        "your workload's answer-length distribution, not a round number."
    )


if __name__ == "__main__":
    main()
