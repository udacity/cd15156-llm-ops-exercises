# TODO(m20-exercise-2): calibration sweep — run a ten-row golden cohort past the
# LLM-judge, tabulate the confusion matrix, and repeat under one varied knob.
# Complete this scaffold. The finished reference lives in solution/ under the same
# marker: grep it there if you get stuck.
"""Calibrate the LLM-judge hallucination check on scikit-learn API correctness.

Fires a ten-query golden cohort through ``route_query`` (five grounded answers,
five that push the model toward a made-up API), scores each with
``check_hallucination``, and prints a confusion matrix plus FP/FN rates. Runs the
whole cohort once per judge model so you can read two matrices side by side and
pick an operating point.

Needs a live ``OPENAI_API_KEY`` (or Vocareum ``voc-`` key) and a loaded corpus
(``make load-data``); every row is a real model round-trip. Ten rows is an
underpowered sample by design — the point is the procedure, not the numbers.

Usage:
    PYTHONPATH=. uv run python scripts/judge_calibration_sweep.py
    PYTHONPATH=. uv run python scripts/judge_calibration_sweep.py --models gpt-4o-mini gpt-4o
"""

import argparse

from src import constants
from src.gateway.router import route_query
from src.guardrails.llm_judge.output_guards import check_hallucination

# Exercise 2: build the two cohorts. Five grounded questions whose answer is fully
# supported by the retrieved chunks (true-negatives the judge should NOT flag), and
# five that name a scikit-learn API that does not exist (true-positives it SHOULD
# flag). See INSTRUCTIONS.md Exercise 2 for example questions.
GROUNDED: list[str] = []
WRONG_API: list[str] = []


def run_cohort(model: str) -> dict[str, int]:
    """Score both cohorts with ``model`` as the judge and return the matrix."""
    # Exercise 2: for each WRONG_API question, route_query then check_hallucination;
    # count TP (judge flagged) vs FN (judge passed). For each GROUNDED question,
    # count TN (passed) vs FP (flagged). Return {"tp", "fn", "tn", "fp"}.
    # Print a line per row (the run makes ~20 live calls and otherwise looks hung).
    raise NotImplementedError("Complete run_cohort — see INSTRUCTIONS.md Exercise 2")


def print_matrix(model: str, m: dict[str, int]) -> None:
    fp_rate = m["fp"] / (m["fp"] + m["tn"]) if (m["fp"] + m["tn"]) else 0.0
    fn_rate = m["fn"] / (m["fn"] + m["tp"]) if (m["fn"] + m["tp"]) else 0.0
    print(f"\n=== judge model: {model} ===")
    print(f"  TP (caught wrong API)   {m['tp']:>2}    FN (missed wrong API) {m['fn']:>2}")
    print(f"  TN (left grounded be)   {m['tn']:>2}    FP (flagged grounded) {m['fp']:>2}")
    print(f"  FP rate = {fp_rate:.2f}   FN rate = {fn_rate:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4o-mini", "gpt-4o"],
        help="Judge models to sweep; the cohort runs once per model.",
    )
    args = parser.parse_args()

    # Exercise 2: run the cohort once per model, varying the judge model
    # (check_hallucination reads constants.MODEL_SIMPLE at call time, so swap it
    # here and restore the original afterward). Print a matrix per model.
    raise NotImplementedError("Complete main — see INSTRUCTIONS.md Exercise 2")


if __name__ == "__main__":
    main()
