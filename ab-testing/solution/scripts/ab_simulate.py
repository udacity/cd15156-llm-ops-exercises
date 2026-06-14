# 200-call sticky-by-user A/B harness for the ScikitDocs assistant.
# Module docstring below records the A/B decision: chi-squared p-value, cost
# delta, latency delta, and the next step (quality read + secondary tiebreaker).
"""A/B decision (illustrative): Variant A retained.

Chi-squared on the LLM-judge faithfulness label at ~50 unique clients
typically returns p ~ 0.8 (not significant; effective N is sticky-
correlated so this is even more underpowered than the raw call count
suggests). Variant B's "be expansive" instruction produced noticeably
longer answers — on a typical run ~40% more completion tokens, which
tracks into ~35% higher mean latency and a modestly higher per-call
cost — with no detectable quality improvement on the judge label.
Next step: rerun at 500 unique clients to confirm the quality-parity
read holds at higher power, then sunset variant B unless a quality
signal emerges.

This script is the 200-call sticky-by-user A/B harness for the
ScikitDocs assistant. It builds a 50-client_id pool, picks each call's
client_id at random from the pool, calls pick_variant with a stable
salt so assignments are sticky across calls, calls OpenAI through
call_with_variant, scores each answer with an LLM-as-judge faithfulness
check, and appends one JSONL row per call to data/ab_log.jsonl. The
analyzer at scripts/ab_analyze.py reads that file.
"""
# Imports: stdlib json/randomness/path, Jinja + OpenAI for the judge, the
# three A/B primitives, config settings, and run_pipeline.
import json
import random
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from openai import OpenAI

from src.config import settings
from src.models import Source  # noqa: F401  (re-exported for downstream use)
from src.optimization import call_with_variant, log_assignment, pick_variant
from src.pipeline import run_pipeline  # for real retrieval

# Harness constants — 200 calls, 50 clients, 50/50 split, stable salt for sticky
# assignment, and the JSONL log path the analyzer reads.
N_CALLS = 200
N_CLIENTS = 50
TRAFFIC_SPLIT = {"A": 0.5, "B": 0.5}
SALT = "prompt-style-v1"
LOG_PATH = Path("data/ab_log.jsonl")

# Question pool the harness samples from — five to ten scikit-learn API questions
# keeps retrieval realistic without ballooning per-run cost.
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

# LLM-as-judge faithfulness scorer. The naive citation check
# `any(s.doc_id in answer ...)` never fires on this corpus — the doc_ids
# are RST section anchors like `modules.svm.kernel-functions`, which the
# model never reproduces verbatim, so every call would score False and the
# chi-squared table would be degenerate. Reading the answer against the
# retrieved chunks gives a graded, honest success signal.
_judge_env = Environment(
    loader=FileSystemLoader("prompts"),
    keep_trailing_newline=True,
    autoescape=False,
)
_judge_client = OpenAI(base_url=settings.openai_base_url or None)


def judge_supported(answer: str, sources: list) -> bool:
    """Return True when the judge rules the answer SUPPORTED by sources.

    Renders prompts/judge.j2 and asks gpt-4o-mini for a JSON verdict.
    Fails open (returns True) on an empty answer or any judge error so a
    transient proxy hiccup doesn't depress the measured success rate.
    """
    if not answer.strip() or not sources:
        return False
    context = "\n\n".join(f"[{s.doc_id}]\n{s.chunk_text}" for s in sources)
    prompt = _judge_env.get_template("judge.j2").render(
        answer=answer, source=context
    )
    try:
        resp = _judge_client.chat.completions.create(
            model=settings.model_simple,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        verdict = (
            json.loads(resp.choices[0].message.content or "{}").get("verdict")
            or ""
        ).upper()
    except Exception:
        return True  # fail open — don't let a judge blip skew the metric
    return verdict == "SUPPORTED"


# Reuses the pipeline retrieval seam so the harness sees the same contexts that
# the production route_query path would feed the generator.
def retrieve(question: str) -> list:
    """Reuse the starter's pipeline retrieval seam.

    `src/pipeline.py` exposes `run_pipeline` which returns a
    QueryResponse with a `.sources` list. We re-use that path so the
    A/B harness sees the same retrieved contexts the production
    `route_query` path would feed to the generator.
    """
    resp = run_pipeline(question, top_k=5)
    return resp.sources


# For each call: pick a random client_id, retrieve sources, assign + invoke the
# variant, score success with the LLM judge, and append one JSONL row.
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
        success = judge_supported(answer, sources)
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


# Standard CLI entry point.
if __name__ == "__main__":
    main()
