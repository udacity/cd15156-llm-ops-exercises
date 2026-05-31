"""CLI: ``make eval`` → ``uv run python scripts/run_eval.py`` (REQ-068, M11).

Runs RAGAS over the ScikitDocs golden set, prints the four-metric
aggregate plus the deprecated-API sub-metric, and optionally writes
per-row results (RAGAS columns + deprecated-API citations) to JSON
for the M11 Exercise 3 diagnostic loop.
"""

import argparse
import json
import sys
from pathlib import Path

from src.evaluation.run_eval import (
    evaluate_pipeline,
    load_golden_set,
    score_deprecated_apis_per_row,
    summarize,
)
from src.evaluation.deprecated_apis import aggregate as aggregate_deprecated

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "data" / "golden_set.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on the ScikitDocs RAG pipeline."
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=GOLDEN_PATH,
        help=f"Golden set CSV (default: {GOLDEN_PATH}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N questions (default: all).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to write per-row results to.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=(
            "Cap RAGAS executor concurrency. Default lets RAGAS use 16. "
            "Set to 1 when running through a contended proxy (Vocareum) "
            "to avoid parallel-load timeouts that produce NaN cells."
        ),
    )
    # TODO(m11-exercise-4)-start
    parser.add_argument(
        "--faithfulness-min",
        type=float,
        default=None,
        help=(
            "If set, exit with code 2 when aggregate faithfulness falls "
            "below this floor. Suitable for CI regression gates."
        ),
    )
    parser.add_argument(
        "--context-recall-min",
        type=float,
        default=None,
        help=(
            "If set, exit with code 2 when aggregate context_recall falls "
            "below this floor. Suitable for CI regression gates."
        ),
    )
    # TODO(m11-exercise-4)-end
    args = parser.parse_args(argv)

    golden = load_golden_set(args.golden)
    if args.limit is not None:
        golden = golden[: args.limit]

    print(f"Evaluating {len(golden)} questions...")
    result = evaluate_pipeline(golden, max_workers=args.max_workers)
    aggregate = summarize(result)

    rows = result.to_pandas().to_dict("records")
    deprecated_rows = score_deprecated_apis_per_row([r["answer"] for r in rows])
    aggregate["deprecated_apis"] = aggregate_deprecated(
        r["score"] for r in deprecated_rows
    )

    print("\nAggregate metrics:")
    for metric, score in aggregate.items():
        print(f"  {metric}: {score:.3f}")

    if args.output:
        for row, deprecated in zip(rows, deprecated_rows):
            row["deprecated_apis_score"] = deprecated["score"]
            row["deprecated_apis_citations"] = deprecated["citations"]
        args.output.write_text(
            json.dumps(
                {"aggregate": aggregate, "rows": rows},
                indent=2,
                default=str,
            )
        )
        print(f"\nWrote per-row results to {args.output}")

    # TODO(m11-exercise-4)-start
    thresholds = {
        "faithfulness": args.faithfulness_min,
        "context_recall": args.context_recall_min,
    }
    for metric, floor in thresholds.items():
        if floor is None:
            continue
        actual = aggregate.get(metric)
        if actual is None or actual < floor:
            print(
                f"FAIL: {metric}={actual!r} below floor {floor}",
                file=sys.stderr,
            )
            sys.exit(2)
    # TODO(m11-exercise-4)-end

    return 0


if __name__ == "__main__":
    sys.exit(main())
