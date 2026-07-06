"""Probe the classifier's tier routing for Module 18 Exercise 1.

Runs the five sample questions from the exercise through the gateway's
``classify`` + ``select_model`` and prints the tier each landed on next to
the concrete model it dispatches to. No server, cache, or cost log involved
— this exercises exactly the two functions Exercise 1 edits.

Because the ``premium`` placeholder is also ``gpt-4o`` (Vocareum-compatible),
the tier is the observable signal, not the model name. This table makes the
tier explicit so ``premium`` is visible even though its model matches
``complex``.

Run via ``make probe-tiers`` (sets PYTHONPATH) or:

    PYTHONPATH=. uv run python scripts/probe_tiers.py

Needs a live ``OPENAI_API_KEY`` — ``classify`` calls gpt-4o-mini in JSON
mode. Classification is not deterministic, so a borderline query may land on
a different tier across runs.
"""

from src.gateway.classifier import classify
from src.gateway.router import select_model

QUESTIONS = [
    "What is the default criterion for RandomForestRegressor?",
    "Compare GradientBoostingClassifier and RandomForestClassifier for imbalanced binary classification.",
    "Walk me through choosing between l1 and l2 penalty on LogisticRegression for a sparse-feature problem.",
    "What changed in StandardScaler between scikit-learn 0.24 and 1.4, and which arguments were deprecated?",
    "Explain every parameter of GridSearchCV's __init__ and how scoring interacts with refit when multiple scorers are passed.",
]


def main() -> None:
    print(f"{'tier':8} | {'model':12} | question")
    print("-" * 78)
    for question in QUESTIONS:
        tier = classify(question)
        print(f"{tier:8} | {select_model(tier):12} | {question[:60]}")


if __name__ == "__main__":
    main()
