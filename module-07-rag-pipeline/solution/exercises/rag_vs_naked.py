"""Exercise 3 — RAG vs naked-call comparison on five questions.

Fires the same five questions through two paths — ``run_pipeline`` (RAG on)
and a direct OpenAI ChatCompletions call (RAG off) — so the learner can tally
where retrieval actually changes the answer. A typical author run is 2 of 5
materially different (LogReg-1.5 version-sensitive, Tokyo population
off-topic) and 3 of 5 materially similar (SVC kernel, HistGradientBoosting,
supervised-vs-unsupervised — parametric memory handles those adequately).

Run from the starter directory::

    uv run python exercises/rag_vs_naked.py | tee /tmp/comparison.txt
"""

# RAG vs naked-call comparison — prints both answers side by side for tallying
# materially-different vs materially-similar outcomes.

from openai import OpenAI

from src.config import settings
from src.pipeline import run_pipeline

# Five questions spanning version-sensitive, recent-API, general-concept, and off-topic axes.
QUESTIONS = [
    "What is the default solver for `LogisticRegression` in scikit-learn 1.5?",
    "What kernel does `SVC` use by default?",
    "What is `HistGradientBoostingClassifier`?",
    "What's the difference between supervised and unsupervised learning?",
    "What's the population of Tokyo?",
]

# Bare OpenAI client sharing the pipeline's key/base_url for the naked path.
client = OpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url or None,
)

# Run each question through both paths and print truncated answers.
for q in QUESTIONS:
    rag = run_pipeline(q)
    naked = client.chat.completions.create(
        model=settings.model_complex,
        messages=[{"role": "user", "content": q}],
    ).choices[0].message.content
    print(f"\n=== Q: {q}")
    print(f"  RAG:   {rag.answer[:200]}")
    print(f"  NAKED: {naked[:200]}")
