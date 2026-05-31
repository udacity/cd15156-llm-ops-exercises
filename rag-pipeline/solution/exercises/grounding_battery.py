"""Exercise 1 — Ten-question grounding battery.

Fires five in-domain factual questions and five off-topic questions through
``run_pipeline`` and prints answer / top source / confidence for each. The
learner classifies each row as grounded / partial / hallucinated and tallies
the result (see SOLUTION_NOTES.md for a representative classification).

Run from the starter directory::

    uv run python exercises/grounding_battery.py | tee /tmp/answers.txt
"""

from src.pipeline import run_pipeline

QUESTIONS = [
    ("in-domain factual",  "What is the default value of `n_estimators` in `RandomForestClassifier`?"),
    ("in-domain factual",  "What does `StandardScaler` do to input features?"),
    ("in-domain factual",  "What kernel does `SVC` use by default?"),
    ("in-domain factual",  "What is the default value of `n_clusters` in `KMeans`?"),
    ("in-domain factual",  "What is the default solver for `LogisticRegression` in scikit-learn 1.5?"),
    ("off-topic benign",   "What's the weather in Paris today?"),
    ("off-topic benign",   "How do I unclog a kitchen sink?"),
    ("off-topic benign",   "Who won the World Cup in 2022?"),
    ("off-topic adjacent", "How do I train a transformer from scratch in PyTorch?"),
    ("off-topic adjacent", "How do I broadcast a 2D array against a 3D array in numpy?"),
]

for category, q in QUESTIONS:
    r = run_pipeline(q)
    print(f"\n=== [{category}] Q: {q}")
    print("A:", r.answer[:280])
    print(f"  top={r.sources[0].doc_id} sim={r.sources[0].similarity_score:.3f}  conf={r.confidence:.3f}")
