# TODO(m22-exercise-1): create scripts/ab_simulate.py — the 200-call sticky-by-user A/B harness.
# TODO(m22-exercise-3): write the A/B decision in the module docstring below — call out
# chi-squared p-value, cost delta, latency delta, and the next step. See INSTRUCTIONS.md
# → Exercise 3 for the required structure (quality read + secondary-metric tiebreaker).
"""A/B decision (illustrative): Variant A retained.

Chi-squared on success rate at 50 unique clients typically returns
p ~ 0.85 (not significant; effective N is sticky-correlated so this
is even more underpowered than the raw call count suggests). Variant
B was ~50% more expensive per call and ~56% slower in mean latency
with no detectable quality improvement on a typical run. Next step:
rerun at 500 unique clients to confirm the quality-parity read holds
at higher power, then sunset variant B unless a quality signal
emerges.

This script is the 200-call sticky-by-user A/B harness for the
ScikitDocs assistant. It builds a 50-client_id pool, picks each
call's client_id at random from the pool, calls pick_variant with a
stable salt so assignments are sticky across calls, calls OpenAI
through call_with_variant, and appends one JSONL row per call to
data/ab_log.jsonl. The analyzer at scripts/ab_analyze.py reads that
file.
"""
# TODO(m22-exercise-1): import random + pathlib + the three A/B primitives + run_pipeline.
import random
from pathlib import Path

from src.models import Source  # noqa: F401  (re-exported for downstream use)
from src.optimization import call_with_variant, log_assignment, pick_variant
from src.pipeline import run_pipeline  # for real retrieval

# TODO(m22-exercise-1): set the harness constants — 200 calls, 50 clients, 50/50 split,
# a stable salt for sticky assignment, and the JSONL log path the analyzer reads.
N_CALLS = 200
N_CLIENTS = 50
TRAFFIC_SPLIT = {"A": 0.5, "B": 0.5}
SALT = "prompt-style-v1"
LOG_PATH = Path("data/ab_log.jsonl")

# TODO(m22-exercise-1): build a small question pool the harness samples from.
# Five to ten scikit-learn API questions is enough to keep retrieval realistic
# without making the run cost balloon.
QUESTIONS = [
    "What is the default criterion for RandomForestClassifier?",
    "How does HistGradientBoostingRegressor handle missing values?",
    "What are the supported solvers for LogisticRegression?",
    "Does DBSCAN require the number of clusters as input?",
    "What's the difference between fit_transform and transform?",
    "How does KMeans pick initial centroids by default?",
    "What metrics does cross_val_score support out of the box?",
    "How do you handle imbalanced classes in scikit-learn?",
    "What is the difference between Pipeline and ColumnTransformer?",
    "How does GridSearchCV decide which combination is best?",
]


# TODO(m22-exercise-1): retrieve() — reuse the starter's pipeline retrieval seam so the
# A/B harness sees the same contexts the production route_query would feed the generator.
def retrieve(question: str) -> list:
    """Reuse the starter's pipeline retrieval seam.

    `src/pipeline.py` exposes `run_pipeline` which returns a
    QueryResponse with a `.sources` list. We re-use that path so the
    A/B harness sees the same retrieved contexts the production
    `route_query` path would feed to the generator.
    """
    resp = run_pipeline(question, top_k=5)
    return resp.sources


# TODO(m22-exercise-1): main() — for each of N_CALLS, pick a random client_id, retrieve
# sources, call pick_variant + call_with_variant, score success as a citation check, and
# append one row to data/ab_log.jsonl via log_assignment.
def main() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    clients = [f"user-{i:03d}" for i in range(N_CLIENTS)]
    for i in range(N_CALLS):
        question = random.choice(QUESTIONS)
        client_id = random.choice(clients)
        sources = retrieve(question)
        variant = pick_variant(client_id, TRAFFIC_SPLIT, salt=SALT)
        answer, usage, cost, latency_ms = call_with_variant(
            question, sources, variant
        )
        source_ids = {s.doc_id for s in sources}
        success = any(sid in answer for sid in source_ids)
        log_assignment(
            LOG_PATH,
            client_id=client_id,
            variant=variant,
            question=question,
            answer=answer,
            usage=usage,
            cost_usd=cost,
            latency_ms=latency_ms,
            success=success,
        )
        if (i + 1) % 20 == 0:
            print(f"{i+1}/{N_CALLS} done")


# TODO(m22-exercise-1): standard CLI entry point.
if __name__ == "__main__":
    main()
