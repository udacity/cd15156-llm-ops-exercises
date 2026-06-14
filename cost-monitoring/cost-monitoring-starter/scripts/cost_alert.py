"""Exercise 3 Part A — rolling-baseline anomaly alert.

Reads ``data/cost_log.jsonl``, computes today's cost, computes the rolling
seven-day average excluding today (skipping zero-cost days), and prints a
warning if today's cost is more than twice the rolling average.

Run:
    uv run python scripts/cost_alert.py
"""

# TODO(m13-ex3): aggregate data/cost_log.jsonl by UTC day (skip blank lines),
# compute today's cost and the rolling 7-day average over the prior days
# (excluding zero-cost days), print both, then print a WARNING line when
# today's cost exceeds 2x the rolling average. See INSTRUCTIONS.md →
# Exercise 3 Part A for the expected output format.
raise NotImplementedError("TODO(m13-ex3)")
