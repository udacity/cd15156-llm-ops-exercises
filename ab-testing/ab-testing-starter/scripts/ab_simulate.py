"""200-call sticky-by-user A/B harness for the ScikitDocs assistant.

Builds a 50-client_id pool, picks each call's client_id at random
from the pool, calls pick_variant with a stable salt so assignments
are sticky across calls, calls OpenAI through call_with_variant, and
appends one JSONL row per call to data/ab_log.jsonl. The analyzer at
scripts/ab_analyze.py reads that file.

When you finish Exercise 2, replace this docstring with the written
A/B decision Exercise 3 asks for — chi-squared p-value, cost delta,
latency delta, and the next step. See INSTRUCTIONS.md.
"""
# TODO(m22-exercise-1): write the 200-call sticky-by-user A/B harness here.
# See INSTRUCTIONS.md → Exercise 1 step 2 for the structure (constants,
# QUESTIONS pool, retrieve() helper, main() loop). After Exercise 2 analysis,
# replace the docstring above with the Exercise 3 decision writeup.
raise NotImplementedError("TODO(m22-exercise-1): implement the A/B harness")
