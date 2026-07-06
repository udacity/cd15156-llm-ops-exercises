# TODO(m20-exercise-3): cost-amplification measurement — fire a baseline cohort and
# a long-generation cohort uncapped, then derive an output cap from the observed
# answer-length distribution (mean + N*sd) and re-fire to show it clip the tail.
# Complete this scaffold (Exercise 3, Step 1); the finished reference lives in
# solution/ under the same marker.
"""Measure the LLM10 cost-amplification surface and set a data-driven output cap.

Fire cohorts through the traced pipeline with a fixed model, in three passes:

  1. baseline              — ordinary questions (normal answer shape)
  2. long, uncapped        — exhaustive-tutorial prompts with max_tokens=None
  3. long, capped          — the same prompts capped at mean + (--sd)*sd of pass 2

Pass 2 measures how long the answers actually run; pass 3 sets the cap from that
distribution so it only bites the tail, the way you would in production. Print
median and p95 cost per cohort, the amplification factor, and which rows clipped.

Call traced_pipeline directly (the gateway's entry): it needs no running server,
but it does need a live OPENAI_API_KEY and a loaded corpus (make load-data).

Usage:
    PYTHONPATH=. uv run python scripts/cost_amplification.py
    PYTHONPATH=. uv run python scripts/cost_amplification.py --sd 0.5
"""

import argparse
import math
import statistics

from src.tracing import traced_pipeline

MODEL = "gpt-4o-mini"

# Exercise 3, Step 1: build the two cohorts. BASELINE is ordinary short questions
# (normal answer shape); LONG asks for exhaustive tutorials that push the model to
# generate a large answer ("Write a thorough tutorial covering every parameter of
# RandomForestClassifier, with worked code examples for each one.").
BASELINE: list[str] = []
LONG: list[str] = []


def measure(questions: list[str], max_tokens: int | None, label: str) -> list[tuple[int, float]]:
    """Fire one cohort at a fixed cap, printing per-row (tokens, cost). Returns them."""
    # Exercise 3, Step 1: for each question call
    # traced_pipeline(question, model=MODEL, max_tokens=max_tokens); collect
    # (resp.tokens.completion_tokens, resp.cost_usd); print a per-row line and flag
    # any row where tokens >= max_tokens (the cap clipped it).
    raise NotImplementedError("Complete measure — see INSTRUCTIONS.md Exercise 3")


def p95(values: list[float]) -> float:
    s = sorted(values)
    k = math.ceil(0.95 * len(s)) - 1
    return s[max(0, min(k, len(s) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10, help="rows per cohort (1-10)")
    parser.add_argument("--sd", type=float, default=1.0,
                        help="cap = mean + this many sd of the uncapped run (lower = more clipping)")
    args = parser.parse_args()

    # Exercise 3, Step 1: measure baseline (uncapped) and long uncapped. Take the
    # mean and stdev of the uncapped completion-token counts (statistics.mean /
    # statistics.stdev), set cap = round(mean + args.sd * sd), then measure the long
    # cohort capped at that value. Print median + p95 per cohort, the amplification
    # factor = long p95 / baseline p95, and how many rows the cap clipped.
    raise NotImplementedError("Complete main — see INSTRUCTIONS.md Exercise 3")


if __name__ == "__main__":
    main()
