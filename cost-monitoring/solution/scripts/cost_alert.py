"""Exercise 3 Part A — rolling-baseline anomaly alert.

Reads ``data/cost_log.jsonl``, computes today's cost, computes the rolling
seven-day average excluding today (skipping zero-cost days), and prints a
warning if today's cost is more than twice the rolling average.

Run:
    uv run python scripts/cost_alert.py
"""

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone

by_day: dict[str, float] = defaultdict(float)
with open("data/cost_log.jsonl") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        day = r["timestamp"][:10]
        by_day[day] += r["cost_usd"]

today = datetime.now(timezone.utc).date().isoformat()
today_cost = by_day.get(today, 0.0)
prior_days = [
    (datetime.fromisoformat(today) - timedelta(days=i)).date().isoformat()
    for i in range(1, 8)
]
prior_costs = [by_day.get(d, 0.0) for d in prior_days]
prior_costs = [c for c in prior_costs if c > 0]
rolling_avg = sum(prior_costs) / len(prior_costs) if prior_costs else 0.0

print(f"Today ({today}): ${today_cost:.4f}")
print(f"Rolling 7d avg (excl today): ${rolling_avg:.4f}")
if rolling_avg > 0 and today_cost > 2 * rolling_avg:
    print(
        f"WARNING: today's cost is {today_cost / rolling_avg:.1f}x the rolling average"
    )
