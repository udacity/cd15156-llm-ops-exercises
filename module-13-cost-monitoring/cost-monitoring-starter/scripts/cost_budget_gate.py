"""Exercise 3 Part B — pre-call tiktoken budget gate.

Defines ``estimate_cost`` (tiktoken-based pre-call estimate) and ``gate``
(raises ``ValueError`` when the estimate exceeds a per-request limit).
Running this module as a script demos both functions on a normal question
and a pathological one, then runs the reconciliation against ``run_pipeline``.

Run:
    uv run python scripts/cost_budget_gate.py
"""

# TODO(m13-ex3): implement estimate_cost(question, model, system_prompt_tokens=1200,
# expected_output_tokens=200) using tiktoken.encoding_for_model + the MODEL_PRICING
# rate table, plus gate(question, model, limit_usd=0.01) which raises ValueError
# when the estimate exceeds limit_usd. Under __main__, demo both on a normal and
# pathological prompt, then run the reconciliation (estimate vs compute_cost on
# run_pipeline output). See INSTRUCTIONS.md → Exercise 3 Part B for the contract
# and expected output format.
raise NotImplementedError("TODO(m13-ex3)")
