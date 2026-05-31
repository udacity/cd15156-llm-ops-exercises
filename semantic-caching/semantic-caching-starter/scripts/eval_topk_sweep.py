"""Sweep ``top_k`` at three values and emit a markdown comparison table.

Sweeps ``top_k ∈ {3, 5, 10}`` by default against the ScikitDocs golden
set. The seeded-difficulty chunks in ``data/seeded_chunks.jsonl``
(REQ-063) guarantee a visible curve — without seeding the corpus is
too clean for a sweep to produce variance.

Cost: ~3× one ``make eval`` invocation (~$0.06 at OpenAI pricing as of
April 2026). Wall-clock: 10-15 minutes at ``max-workers=1`` on
Vocareum, faster on direct endpoints.

Usage:
    set -a; source .env; set +a
    make eval-topk-sweep
    # or
    uv run python scripts/eval_topk_sweep.py --topks 3,5,10
"""

import argparse
import sys
from pathlib import Path

from src.evaluation.run_eval import (
    evaluate_pipeline,
    load_golden_set,
    summarize,
)

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "data" / "golden_set.csv"
DEFAULT_TOPKS = (3, 5, 10)
METRIC_COLUMNS = (
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
)


def _parse_topks(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _format_row(top_k: int, scored: dict[str, float]) -> str:
    cells = [f"{top_k}"]
    for metric in METRIC_COLUMNS:
        value = scored.get(metric)
        cells.append(f"{value:.3f}" if value is not None else "—")
    return "| " + " | ".join(cells) + " |"


def _render_markdown(rows: list[tuple[int, dict[str, float]]]) -> str:
    header = "| top_k | " + " | ".join(METRIC_COLUMNS) + " |"
    sep = "|-------|" + "|".join("-" * (len(m) + 2) for m in METRIC_COLUMNS) + "|"
    body = [_format_row(top_k, scored) for top_k, scored in rows]
    return "\n".join([header, sep, *body])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sweep top_k values and emit a RAGAS comparison table."
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
        help="Run only the first N questions per sweep value (default: all).",
    )
    parser.add_argument(
        "--topks",
        type=_parse_topks,
        default=list(DEFAULT_TOPKS),
        help=f"Comma-separated top_k values to sweep (default: {','.join(map(str, DEFAULT_TOPKS))}).",
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
    args = parser.parse_args(argv)

    golden = load_golden_set(args.golden)
    if args.limit is not None:
        golden = golden[: args.limit]

    rows: list[tuple[int, dict[str, float]]] = []
    for top_k in args.topks:
        print(
            f"==> Sweeping top_k={top_k} over {len(golden)} questions ...",
            file=sys.stderr,
        )
        result = evaluate_pipeline(
            golden, top_k=top_k, max_workers=args.max_workers
        )
        rows.append((top_k, summarize(result)))

    print()
    print(_render_markdown(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
