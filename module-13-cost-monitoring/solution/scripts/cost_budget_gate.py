"""Exercise 3 Part B — pre-call tiktoken budget gate.

Defines ``estimate_cost`` (tiktoken-based pre-call estimate) and ``gate``
(raises ``ValueError`` when the estimate exceeds a per-request limit).
Running this module as a script demos both functions on a normal question
and a pathological one, then runs the reconciliation against ``run_pipeline``.

Run:
    PYTHONPATH=. uv run python scripts/cost_budget_gate.py
"""

# Pre-call tiktoken cost estimator and per-request budget gate, with reconciliation demo
import tiktoken

from src.cost.tracker import compute_cost
from src.pipeline import run_pipeline
from src.pricing import MODEL_PRICING


def estimate_cost(
    question: str,
    model: str,
    system_prompt_tokens: int = 1200,
    expected_output_tokens: int = 200,
) -> float:
    """Return a pre-call USD cost estimate for the given question and model."""
    enc = tiktoken.encoding_for_model(model)
    question_tokens = len(enc.encode(question))
    input_tokens = question_tokens + system_prompt_tokens
    input_rate, output_rate = MODEL_PRICING[model]
    return (
        input_tokens * input_rate + expected_output_tokens * output_rate
    ) / 1_000_000


def gate(question: str, model: str, limit_usd: float = 0.01) -> None:
    """Raise ``ValueError`` if the estimated cost exceeds the per-request limit."""
    est = estimate_cost(question, model)
    if est > limit_usd:
        raise ValueError(
            f"Estimated cost ${est:.4f} exceeds limit ${limit_usd:.4f}"
        )


if __name__ == "__main__":
    # Normal-question gate passes silently.
    normal = "What kernel does SVC use by default?"
    gate(normal, "gpt-4o-mini", 0.01)
    print(f"normal gate passed: ${estimate_cost(normal, 'gpt-4o-mini'):.6f}")

    # Pathological prompt with a tight limit raises.
    pathological = "very long question..." * 1000
    try:
        gate(pathological, "gpt-4o", 0.001)
    except ValueError as e:
        print(f"pathological gate refused: {e}")

    # Reconciliation: estimate vs post-call truth.
    r = run_pipeline(normal)
    actual_cost = compute_cost(r.model, r.tokens)
    est = estimate_cost(normal, r.model)
    delta_pct = ((est - actual_cost) / actual_cost * 100) if actual_cost else 0.0
    print()
    print(f"question: {normal!r}")
    print(f"model:    {r.model}")
    print(f"estimate_cost (system=1200, expected_out=200): ${est:.6f}")
    print(f"actual prompt_tokens:    {r.tokens.prompt_tokens}")
    print(f"actual completion_tokens: {r.tokens.completion_tokens}")
    print(f"compute_cost(actual):    ${actual_cost:.6f}")
    sign = "high" if delta_pct >= 0 else "low"
    print(f"delta: estimate is {abs(delta_pct):.0f}% {sign}")
