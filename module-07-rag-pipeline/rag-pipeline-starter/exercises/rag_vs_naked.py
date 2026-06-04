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

# TODO(m07-ex3): fire the same five questions through run_pipeline (RAG on)
# and a direct OpenAI ChatCompletions call (RAG off), print both answers side
# by side so the learner can tally materially-different vs materially-similar.
raise NotImplementedError("TODO(m07-ex3)")
