"""Run RAGAS evaluation against the ScikitDocs golden test set.

Only the four stable metrics are run by default —
``faithfulness``, ``answer_relevancy``, ``context_recall``,
``context_precision``. Other RAGAS metrics churn between point releases
and have produced inconsistent scores during course development; this
list is the four the starter trusts to compare runs against.

To add a metric, append it to ``DEFAULT_METRICS`` (or pass a custom
list to ``evaluate_pipeline``). ``pyproject.toml`` pins ``ragas==0.4.3``
because 0.x patches break occasionally and the four-metric stack was
validated against that pin.

The starter layers one ScikitDocs-specific sub-metric on top of the
RAGAS four: ``deprecated_apis`` (see ``src/evaluation/deprecated_apis.py``).
It scores whether a generated answer cites any scikit-learn symbol
that is deprecated or removed in the corpus-pinned 1.5 release. The
sub-metric runs outside RAGAS — RAGAS scores the four mainline metrics,
the deprecated-API checker runs row-by-row over the answers, and both
are folded into the per-row JSON output ``scripts/run_eval.py`` writes.
"""

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from datasets import Dataset
from langchain_openai import ChatOpenAI as LangchainChatOpenAI
from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

from src import constants
from src.config import settings
from src.evaluation.deprecated_apis import (
    find_deprecated_citations,
    score_deprecated_apis,
)
from src.pipeline import run_pipeline

DEFAULT_METRICS = [
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
]


def load_golden_set(path: str | Path) -> list[dict]:
    """Load the ScikitDocs golden CSV.

    Returns a list of ``{question, ground_truth, query_type, version_sensitive}``
    dicts. The starter's CSV uses ``ground_truth_answer`` as the column
    name (rather than the capstone's ``ground_truth``); this function
    rewrites it to ``ground_truth`` so callers downstream see the
    RAGAS-canonical key. ``expected_doc_ids`` and ``min_hits`` are
    consumed only by ``scripts/smoke_gate.py`` and are dropped here —
    RAGAS measures retrieval quality against whatever the pipeline
    retrieves at eval time, not against a fixed candidate list.
    """
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "question": row["question"],
                    "ground_truth": row["ground_truth_answer"],
                    "query_type": row.get("query_type", ""),
                    "version_sensitive": row.get("version_sensitive", "").lower()
                    == "true",
                }
            )
    return rows


def build_eval_dataset(golden_set: list[dict], *, top_k: int = 5) -> Dataset:
    """Run the RAG pipeline per row; collect what RAGAS needs.

    ``top_k`` is forwarded to ``run_pipeline`` so ``eval_topk_sweep.py``
    can call this function once per sweep value. The default matches
    ``constants.DEFAULT_TOP_K`` so existing call sites keep their
    current behavior.
    """
    questions: list[str] = []
    answers: list[str] = []
    retrieved_contexts: list[list[str]] = []
    ground_truths: list[str] = []

    for row in golden_set:
        response = run_pipeline(row["question"], top_k=top_k)
        questions.append(row["question"])
        answers.append(response.answer)
        retrieved_contexts.append([s.chunk_text for s in response.sources])
        ground_truths.append(row["ground_truth"])

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": retrieved_contexts,
            "ground_truth": ground_truths,
        }
    )


def _build_llm() -> LangchainLLMWrapper:
    """Build a RAGAS LLM wrapper with two corrections to RAGAS defaults.

    1. ``max_tokens=8192``. RAGAS auto-instantiates a default
       ``ChatOpenAI`` with ``max_tokens`` unset; OpenAI applies its own
       cap (~3072 for ``gpt-4o-mini``), which truncates RAGAS's
       statement-extraction prompts on multi-symbol questions and
       produces ``output incomplete due to a max_tokens length limit``
       errors. ``gpt-4o-mini`` supports 16,384 output tokens; 8k is
       generous-but-not-wasteful.

    2. ``bypass_n=True``. RAGAS's default ``LangchainLLMWrapper.agenerate_text``
       (ragas 0.4.x) mutates the shared ``langchain_llm.n`` attribute
       per call. With ``max_workers=16`` (RAGAS's default), 16
       coroutines race to set ``.n``; the loser sometimes sees ``n=1``,
       the metric gets one generation instead of three, and RAGAS
       warns ``LLM returned 1 generations instead of requested 3``.
       ``bypass_n=True`` routes through the fallback path that issues
       N separate ``n=1`` calls sequentially per row — no shared
       mutation, no race. Roughly ``3×`` the API calls per row, but
       smaller per-call latency on Vocareum, so wall-clock is
       comparable. Eliminates the race entirely.

    Temperature is pinned at ``constants.JUDGE_TEMPERATURE`` (0.0). The
    judge is a comparator, not a generator — non-zero temperature is a
    documented source of run-to-run drift in LLM-as-judge evals (see
    the Module 10 concept walk and the MT-Bench paper).
    """
    return LangchainLLMWrapper(
        LangchainChatOpenAI(
            model=settings.model_simple,
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_base_url or None,
            max_tokens=8192,
            temperature=constants.JUDGE_TEMPERATURE,
        ),
        bypass_n=True,
    )


def _build_embeddings() -> LangchainEmbeddingsWrapper:
    """Build a RAGAS embeddings provider from the starter's OpenAI settings.

    RAGAS auto-instantiates a default embeddings provider when none is
    supplied to ``evaluate(...)``. In 0.4.x that default is
    ``ragas.embeddings.OpenAIEmbeddings`` — the modern provider, which
    exposes ``embed_text`` but not ``embed_query``. The metrics
    (``answer_relevancy``, in particular) still call ``embed_query``
    in 0.4.3, so the modern provider raises ``AttributeError`` per row
    and the metric ends up NaN.

    The deprecated-but-still-functional ``LangchainEmbeddingsWrapper``
    around ``langchain_openai.OpenAIEmbeddings`` exposes ``embed_query``
    and works with every metric in 0.4.3. ``OPENAI_BASE_URL`` is honored
    so Vocareum deployments work without extra config.
    """
    return LangchainEmbeddingsWrapper(
        LangchainOpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_base_url or None,
            model=settings.embedding_model,
        )
    )


def evaluate_pipeline(
    golden_set: list[dict],
    metrics: Sequence | None = None,
    *,
    top_k: int = 5,
    max_workers: int | None = None,
):
    """Run the pipeline against the golden set and score with RAGAS.

    ``top_k`` is forwarded into ``build_eval_dataset`` and from there
    into ``run_pipeline``. ``scripts/eval_topk_sweep.py`` calls this
    once per sweep value.

    ``max_workers`` caps the RAGAS executor concurrency. ``None`` lets
    RAGAS use its built-in default (16). The starter's CLI passes 1 by
    default because the Vocareum proxy throttles parallel judge calls
    into ``TimeoutError`` floods that turn metric averages into NaN.
    """
    dataset = build_eval_dataset(golden_set, top_k=top_k)
    kwargs: dict[str, Any] = {
        "metrics": list(metrics or DEFAULT_METRICS),
        "embeddings": _build_embeddings(),
        "llm": _build_llm(),
    }
    if max_workers is not None:
        kwargs["run_config"] = RunConfig(max_workers=max_workers)
    return evaluate(dataset, **kwargs)


def score_deprecated_apis_per_row(answers: Sequence[str]) -> list[dict]:
    """Run the deprecated-API sub-metric over the answers in a result.

    Returns one dict per answer with ``score`` (1.0 or 0.0) and
    ``citations`` (the symbol names that tripped the check). Callers
    can fold this into the per-row JSON or aggregate it into a single
    mean alongside the four RAGAS metrics.
    """
    rows: list[dict] = []
    for answer in answers:
        citations = find_deprecated_citations(answer)
        rows.append(
            {
                "score": score_deprecated_apis(answer),
                "citations": [api.symbol for api in citations],
            }
        )
    return rows


def summarize(result) -> dict[str, float]:
    """Aggregate per-row metric scores into a flat ``{metric: mean}`` dict.

    RAGAS adds string-typed columns alongside metric scores
    (``user_input``, ``retrieved_contexts``, ``response``, ``reference``
    since 0.2.x). Reduce only numeric columns to avoid ``mean()`` raising
    on strings.
    """
    import pandas as pd

    df = result.to_pandas()
    scored: dict[str, float] = {}
    for col in df.columns:
        if col in {"question", "answer", "contexts", "ground_truth"}:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        scored[col] = float(df[col].mean())
    return scored


__all__ = [
    "DEFAULT_METRICS",
    "build_eval_dataset",
    "evaluate_pipeline",
    "load_golden_set",
    "score_deprecated_apis_per_row",
    "summarize",
]
