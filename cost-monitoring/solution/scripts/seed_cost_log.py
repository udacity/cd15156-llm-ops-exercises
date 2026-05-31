"""Seed ``data/cost_log.jsonl`` with 50 realistic synthetic entries (REQ-069, M13).

Lets a learner satisfy Module 13 Exercise 1's "50-entry floor" in <1
second with $0 of API spend, instead of running a 50-query bash loop.
The mix and token distributions match what a real run produces, so
``GET /cost-dashboard`` looks indistinguishable from a real load test.

Idempotent: if the log already has ≥``--target`` entries, the script
prints a ``skipping`` message and exits 0 without touching the file.
If the log has fewer, it appends until reaching the target and
preserves any pre-existing rows.

Usage:
    make seed-cost-log              # append to 50 entries (no-op if already there)
    uv run python scripts/seed_cost_log.py --reset      # wipe and regenerate
    uv run python scripts/seed_cost_log.py --target 100 # custom target
"""

import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src import constants
from src.models import TokenUsage
from src.pricing import compute_cost

LOG_PATH = Path(constants.COST_LOG_PATH)
DEFAULT_TARGET = 50
SEED = 20260501  # deterministic seed for reproducibility

# Mix proportions per Module 13 framing: 70% simple, 20% complex, 10% LLM-judge.
MIX = (
    ("simple", constants.MODEL_SIMPLE, 0.70),
    ("complex", constants.MODEL_COMPLEX, 0.20),
    ("hallucination_check", constants.MODEL_SIMPLE, 0.10),
)

# Token-count distributions (mean, stddev, [min, max]) — eyeballed from a real
# ScikitDocs run. Prompts cluster ~1100-1300 (docbot system prompt + 5 retrieved
# chunks); completions vary by query type.
TOKEN_DISTRIBUTIONS: dict[str, dict[str, tuple[int, int, int, int]]] = {
    "simple": {
        "prompt": (1150, 50, 1050, 1300),
        "completion": (60, 30, 25, 150),
    },
    "complex": {
        "prompt": (1200, 70, 1100, 1350),
        "completion": (350, 80, 200, 600),
    },
    "hallucination_check": {
        "prompt": (900, 80, 750, 1100),
        "completion": (50, 25, 30, 120),
    },
}


def _sampled_int(rng: random.Random, mean: int, stddev: int, lo: int, hi: int) -> int:
    """Gaussian sample clamped to a sensible range."""
    return max(lo, min(hi, int(rng.gauss(mean, stddev))))


def _existing_count(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _allocate(remaining: int) -> list[tuple[str, str]]:
    """Build a list of (query_type, model) entries totalling ``remaining``,
    preserving the 70/20/10 mix as closely as integer division allows.
    """
    counts = [(qt, model, max(1, round(remaining * pct))) for qt, model, pct in MIX]
    drift = remaining - sum(c for _, _, c in counts)
    counts[0] = (counts[0][0], counts[0][1], counts[0][2] + drift)
    expanded: list[tuple[str, str]] = []
    for qt, model, n in counts:
        expanded.extend([(qt, model)] * max(0, n))
    return expanded[:remaining]


def _synthesize(remaining: int, rng: random.Random, now: datetime) -> list[dict]:
    """Generate ``remaining`` synthetic records spread across the past 24h."""
    plan = _allocate(remaining)
    rng.shuffle(plan)
    records: list[dict] = []
    for i, (query_type, model) in enumerate(plan):
        dist = TOKEN_DISTRIBUTIONS[query_type]
        prompt = _sampled_int(rng, *dist["prompt"])
        completion = _sampled_int(rng, *dist["completion"])
        usage = TokenUsage(prompt_tokens=prompt, completion_tokens=completion)
        offset = timedelta(
            seconds=int(86400 * (i + 1) / (remaining + 1))
            + rng.randint(-120, 120)
        )
        ts = (now - timedelta(hours=24) + offset).isoformat()
        records.append(
            {
                "timestamp": ts,
                "model": model,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "cost_usd": compute_cost(model, usage),
                "query_type": query_type,
            }
        )
    records.sort(key=lambda r: r["timestamp"])
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Seed data/cost_log.jsonl with realistic synthetic entries."
    )
    parser.add_argument(
        "--target", type=int, default=DEFAULT_TARGET,
        help=f"Total entries the log should have after this run (default {DEFAULT_TARGET}).",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Truncate the log before seeding instead of appending.",
    )
    parser.add_argument(
        "--path", type=str, default=str(LOG_PATH),
        help=f"Override the log path (default {LOG_PATH}).",
    )
    args = parser.parse_args(argv)

    log_path = Path(args.path)
    if args.reset and log_path.exists():
        log_path.unlink()

    existing = _existing_count(log_path)
    if existing >= args.target:
        print(
            f"skipping — log already populated ({existing} entries ≥ target {args.target})",
            file=sys.stderr,
        )
        return 0

    remaining = args.target - existing
    rng = random.Random(SEED + existing)
    records = _synthesize(remaining, rng, datetime.now(timezone.utc))

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(
        f"seeded {len(records)} entries "
        f"({existing} existing + {len(records)} new = {existing + len(records)} total)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
