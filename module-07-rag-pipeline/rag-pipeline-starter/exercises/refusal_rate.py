"""Exercise 2 — Refusal-rate measurement before and after a prompt edit.

Run twice: once against the baseline ``prompts/docbot_system.j2`` (instruction
6 is strict), once after replacing instruction 6 with a permissive directive.
Count refusals in each pass; see SOLUTION_NOTES.md in the solution directory
for the recommended edit and a typical baseline=5/5, permissive=2/5 outcome.

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
raise NotImplementedError("TODO(m07-ex2)")
