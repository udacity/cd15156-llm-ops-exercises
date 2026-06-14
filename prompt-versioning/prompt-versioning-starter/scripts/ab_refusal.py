"""A/B-test refusal rate between two prompt variants.

Runs the same N questions against both branches by:
    - `git checkout main`              for variant A
    - `git checkout ex3-soft-refusal`  for variant B
between iterations. Writes the per-question label (refused vs answered)
to a JSON report and prints a chi-squared p-value.
"""
# TODO(m03-ex3): build A/B refusal-rate harness (whole file)
raise NotImplementedError("TODO(m03-ex3): implement A/B refusal harness")
