"""Exercise 2 — Refusal-rate measurement before and after a prompt edit.

Run twice: once against the baseline ``prompts/docbot_system.j2`` (instruction
6 is strict), once after replacing instruction 6 with a permissive directive.
Count refusals in each pass; see SOLUTION_NOTES.md for the recommended edit
and a typical baseline=5/5, permissive=2/5 outcome.

Run from the starter directory::

    # baseline
    uv run python exercises/refusal_rate.py | tee /tmp/baseline-refusal.txt
    # edit prompts/docbot_system.j2 instruction 6, then re-run:
    uv run python exercises/refusal_rate.py | tee /tmp/permissive-refusal.txt
    # revert when done:
    git checkout prompts/docbot_system.j2
"""

# TODO(m07-ex2): fire the same five off-topic questions through run_pipeline
# and print each Q/A pair; run it once against baseline docbot_system.j2 and
# once after softening instruction 6 to measure the refusal-rate delta.

from src.pipeline import run_pipeline

# TODO(m07-ex2): list the five off-topic questions (3 benign + 2 adjacent).
OFF_TOPIC = [
    "What's the weather in Paris today?",
    "How do I unclog a kitchen sink?",
    "Who won the World Cup in 2022?",
    "How do I train a transformer from scratch in PyTorch?",
    "How do I broadcast a 2D array against a 3D array in numpy?",
]

# TODO(m07-ex2): loop the five off-topic questions through run_pipeline and
# print Q/A pairs so the learner can score each as refused vs not-refused.
for q in OFF_TOPIC:
    r = run_pipeline(q)
    print(f"\nQ: {q}\nA: {r.answer[:300]}")
