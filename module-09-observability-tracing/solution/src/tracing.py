"""Phoenix tracing for the ScikitDocs pipeline.

Two responsibilities:

1. ``init_tracing`` boots the embedded Phoenix UI, registers an
   OpenTelemetry ``TracerProvider``, and auto-instruments the OpenAI
   SDK. Idempotent — safe to call multiple times from the lifespan of a
   long-running process or from a one-off Python invocation.

2. ``traced_pipeline`` composes the same four functions
   ``pipeline.run_pipeline`` does — ``embed_query`` → ``store.query`` →
   ``render_system_prompt`` → ``generate`` — and emits one named span per
   stage. The resulting hierarchy:

       rag_query           (root, the request)
         └── retrieve
              ├── embed    (auto-child: OpenAI Embedding span)
              └── search   (Chroma cosine top-k)
         ├── augment       (Jinja system-prompt render)
         └── generate      (auto-child: OpenAI ChatCompletion span)

   Re-implementing the composition here (instead of decorating
   ``run_pipeline``) is deliberate: each stage becomes its own span in
   Phoenix, which is the diagnostic surface the eval, cost, and latency
   layers read from. The trade-off is the small duplication with
   ``src/pipeline.py``, which stays free of any OpenTelemetry import (so it
   remains a pure RAG composition) — keeping the duplication contained to
   this one file.

Trace-export rendering helpers (``summarize_traces``, ``render_markdown``,
``render_json``) live in the same file so the starter sticks to its
one-flat-file-per-module convention. ``scripts/show_traces.py`` is a
thin CLI that calls them.
"""

import functools
import json
import math
import os
import time
from typing import Any, Callable

from opentelemetry import trace as otel_trace
from opentelemetry.trace import Status, StatusCode

from src import constants
from src.config import settings
from src.models import QueryResponse, Source, TokenUsage

# Module-level singletons — ``init_tracing`` is idempotent and uses
# these to short-circuit on repeat calls. Tests that mock the phoenix
# import reset them via ``_reset_for_tests()`` below.
_tracer_provider: Any = None
_phoenix_session: Any = None


def init_tracing() -> None:
    """Launch embedded Phoenix and register the OpenAI auto-instrumentor.

    Idempotent. ``settings.tracing_backend == "none"`` short-circuits the
    whole function — the test suite uses this kill-switch.

    Side effects on first call:

    - ``phoenix.launch_app()`` starts the Phoenix UI bound to
      ``settings.phoenix_host:phoenix_port`` (defaults
      ``0.0.0.0:6006`` from ``constants.PHOENIX_PORT``). The
      ``PHOENIX_WORKING_DIR``, ``PHOENIX_HOST``, ``PHOENIX_PORT``
      env vars are ``setdefault`` so an explicit override (e.g.
      ``PHOENIX_HOST_ROOT_PATH`` from ``make serve-proxy``) wins.
    - ``phoenix.otel.register(...)`` configures an OTLP exporter
      pointed at that Phoenix instance under the project name
      ``constants.PHOENIX_PROJECT_NAME`` (``"scikitdocs"``).
    - ``OpenAIInstrumentor().instrument(...)`` patches the OpenAI SDK so
      every chat-completions and embeddings call emits an OpenInference
      span — no per-call instrumentation needed in ``embedder`` or
      ``generator``.
    """
    global _tracer_provider, _phoenix_session

    if settings.tracing_backend == "none":
        return

    if settings.phoenix_embedded and _phoenix_session is None:
        import phoenix as px

        os.environ.setdefault("PHOENIX_WORKING_DIR", settings.phoenix_working_dir)
        os.environ.setdefault("PHOENIX_HOST", settings.phoenix_host)
        os.environ.setdefault("PHOENIX_PORT", str(settings.phoenix_port))
        _phoenix_session = px.launch_app()

    if _tracer_provider is None:
        from openinference.instrumentation.openai import OpenAIInstrumentor
        from phoenix.otel import register

        endpoint = (
            f"http://{settings.phoenix_host}:{settings.phoenix_port}/v1/traces"
        )
        _tracer_provider = register(
            project_name=settings.phoenix_project_name,
            endpoint=endpoint,
            verbose=False,
        )
        OpenAIInstrumentor().instrument(tracer_provider=_tracer_provider)


def flush() -> None:
    """Force-flush queued spans. Safe to call on a process-shutdown path."""
    if _tracer_provider is not None:
        try:
            _tracer_provider.force_flush()
        except Exception:
            # Shutdown path — never let a flush failure crash the host.
            pass


def _set_str_attr(span: Any, key: str, value: Any) -> None:
    """OTel attributes must be primitives — coerce non-primitives to str."""
    if isinstance(value, (str, int, float, bool)):
        span.set_attribute(key, value)
    else:
        span.set_attribute(key, str(value))


def _summarize_sources(sources: list[Source]) -> str:
    """Compact ``rag.sources`` attribute — list of ``(doc_id, score)`` pairs."""
    return str([(s.doc_id, round(s.similarity_score, 3)) for s in sources])


def traced_pipeline(
    question: str, model: str | None = None, top_k: int = 5
) -> QueryResponse:
    """Run ``run_pipeline``'s composition with explicit spans per stage.

    Imports the four underlying functions directly (rather than calling
    ``run_pipeline``) so ``retrieve``, ``embed``, ``search``, ``augment``,
    and ``generate`` each become their own named span, so the resulting
    Gantt chart in Phoenix shows one row per pipeline stage.

    Returns a ``QueryResponse`` with ``trace_id`` populated (32-character
    hex from the W3C Trace Context spec). When ``init_tracing`` was not
    called or the backend is ``"none"``, the OTel API returns a no-op
    tracer; spans are silently dropped and ``trace_id`` is ``None``.
    """
    # Imports here (not at module top): the tracer wrapper composes the
    # pipeline by hand, but importing the underlying functions lazily means
    # ``tracing.py`` drops into a fresh checkout without circular-import
    # gymnastics.
    from src.embedder import embed_query
    from src.generator import generate, render_system_prompt
    from src.store import query

    chosen_model = model or settings.model_complex
    tracer = otel_trace.get_tracer("src.tracing")

    with tracer.start_as_current_span("rag_query") as span:
        span.set_attribute("input.value", question)
        _set_str_attr(span, "rag.input.model", chosen_model)
        _set_str_attr(span, "rag.input.top_k", top_k)

        start = time.perf_counter()
        try:
            # === retrieve = embed + search ===
            with tracer.start_as_current_span("retrieve") as retrieve_span:
                retrieve_span.set_attribute("rag.top_k", top_k)
                with tracer.start_as_current_span("embed") as embed_span:
                    embed_span.set_attribute("rag.embedding_model", settings.embedding_model)
                    query_embedding = embed_query(question)
                    embed_span.set_attribute("rag.embedding_dim", len(query_embedding))
                with tracer.start_as_current_span("search") as search_span:
                    search_span.set_attribute("rag.top_k", top_k)
                    sources = query(query_embedding, n_results=top_k)
                    search_span.set_attribute("rag.sources.count", len(sources))
                    if sources:
                        search_span.set_attribute(
                            "rag.sources.top_score", max(s.similarity_score for s in sources)
                        )
                # Record top similarity score on the outer retrieve span for slice-by-score analysis.
                retrieve_span.set_attribute(
                    "rag.retrieve.top_score",
                    max(s.similarity_score for s in sources) if sources else 0.0,
                )
                retrieve_span.set_attribute("rag.sources.count", len(sources))

            # === augment = render system prompt ===
            with tracer.start_as_current_span("augment") as augment_span:
                system_prompt = render_system_prompt(sources)
                augment_span.set_attribute(
                    "rag.system_prompt.length_chars", len(system_prompt)
                )

            # === generate = OpenAI chat completions ===
            with tracer.start_as_current_span("generate") as gen_span:
                gen_span.set_attribute("llm.model_name", chosen_model)
                gen_span.set_attribute("input.value", question)
                answer, usage, cost = generate(question, sources, chosen_model)
                gen_span.set_attribute("output.value", answer)
                gen_span.set_attribute("llm.token_count.prompt", usage.prompt_tokens)
                gen_span.set_attribute(
                    "llm.token_count.completion", usage.completion_tokens
                )
                gen_span.set_attribute("llm.token_count.total", usage.total)
                gen_span.set_attribute("rag.cost_usd", cost)

        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            raise

        latency_ms = (time.perf_counter() - start) * 1000.0
        confidence = (
            sum(s.similarity_score for s in sources) / len(sources) if sources else 0.0
        )

        # === root span attributes ===
        span.set_attribute("output.value", answer)
        span.set_attribute("rag.latency_ms", latency_ms)
        span.set_attribute("rag.confidence", confidence)
        span.set_attribute("rag.cost_usd", cost)
        span.set_attribute("rag.top_k", len(sources))
        span.set_attribute("rag.model", chosen_model)
        _set_str_attr(span, "rag.sources", _summarize_sources(sources))

        ctx = span.get_span_context()
        trace_id = format(ctx.trace_id, "032x") if ctx and ctx.is_valid else None

        return QueryResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            model=chosen_model,
            tokens=usage,
            cost_usd=cost,
            trace_id=trace_id,
        )


# === Trace export — markdown + JSON renderers ===
#
# Kept in this file (not a separate module) so the starter sticks to its
# "one flat file per module" convention. ``scripts/show_traces.py``
# is the thin CLI that calls these.


def _safe_int(value: Any) -> int:
    """Coerce a possibly-NaN/None pandas scalar to int. Defaults to 0.

    Phoenix attaches token counts only to LLM spans; retriever and root
    spans often have NaN here. ``int(NaN)`` raises, and ``NaN or 0``
    doesn't short-circuit (NaN is truthy), so guard explicitly.
    """
    if value is None:
        return 0
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0
    if math.isnan(f):
        return 0
    return int(f)


def summarize_traces(df: Any, last_n: int) -> list[dict]:
    """Group spans by ``trace_id`` and return one summary dict per trace.

    Each summary carries the rubric §7 fields: trace_id (truncated for
    display), question, model, latency_ms, prompt/completion tokens,
    and the slowest child span name + duration.
    """
    if df is None or len(df) == 0:
        return []

    summaries: list[dict] = []
    for trace_id, group in df.groupby("context.trace_id"):
        root = group[group["name"] == "rag_query"]
        if len(root) == 0:
            root = group[group["parent_id"].isna()]
        if len(root) == 0:
            continue
        root_row = root.iloc[0]

        children = group[group["name"] != "rag_query"]
        if len(children) > 0:
            children = children.copy()
            children["_dur_ms"] = (
                (children["end_time"] - children["start_time"]).dt.total_seconds()
                * 1000
            )
            slowest = children.loc[children["_dur_ms"].idxmax()]
            slowest_name = str(slowest["name"])
            slowest_ms = round(float(slowest["_dur_ms"]), 1)
        else:
            slowest_name = "—"
            slowest_ms = 0.0

        # Phoenix flattens span attributes with namespaced keys like
        # `attributes.rag.<key>` into a single dict-valued column
        # `attributes.rag` — NOT into one flat column per nested key.
        rag_attrs = root_row.get("attributes.rag") or {}
        if not isinstance(rag_attrs, dict):
            rag_attrs = {}

        # Token counts live on the ``generate`` child (set above), not
        # on the root.
        gen = group[group["name"] == "generate"]
        if len(gen) > 0:
            gen_row = gen.iloc[0]
            prompt_tok = _safe_int(gen_row.get("attributes.llm.token_count.prompt", 0))
            completion_tok = _safe_int(
                gen_row.get("attributes.llm.token_count.completion", 0)
            )
        else:
            prompt_tok = 0
            completion_tok = 0

        summaries.append(
            {
                "trace_id": str(trace_id)[:8],
                "question": str(root_row.get("attributes.input.value", "—"))[:80],
                "model": str(rag_attrs.get("model", "—")),
                "latency_ms": round(float(rag_attrs.get("latency_ms", 0) or 0), 1),
                "prompt_tokens": prompt_tok,
                "completion_tokens": completion_tok,
                "slowest_span": slowest_name,
                "slowest_ms": slowest_ms,
                "start_time": str(root_row.get("start_time", "")),
            }
        )

    summaries.sort(key=lambda s: s["start_time"], reverse=True)
    return summaries[:last_n]


def render_markdown(summaries: list[dict], total: int) -> str:
    """Render the per-trace summary as a markdown table."""
    if not summaries:
        return (
            "# Phoenix Trace Export\n\n"
            "No traces found yet — fire a few queries via `traced_pipeline`.\n"
        )

    lines = [
        "# Phoenix Trace Export",
        "",
        f"{total} trace(s) captured. Showing the most recent {len(summaries)}.",
        "",
        (
            "| # | Trace ID | Question | Model | Latency (ms) | "
            "Prompt tok | Compl. tok | Slowest child | Slowest (ms) |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, s in enumerate(summaries, 1):
        lines.append(
            f"| {i} | `{s['trace_id']}` | {s['question']} | "
            f"{s['model']} | {s['latency_ms']} | "
            f"{s['prompt_tokens']} | {s['completion_tokens']} | "
            f"{s['slowest_span']} | {s['slowest_ms']} |"
        )
    return "\n".join(lines) + "\n"


def render_json(summaries: list[dict]) -> str:
    """Serialise summaries as pretty-printed JSON."""
    return json.dumps(summaries, indent=2, default=str)


def render_spans_json(df: Any, last_n: int) -> str:
    """Dump raw spans (name + ``attributes.rag``) for the most-recent traces.

    Unlike ``summarize_traces`` — which collapses each trace to one root
    row — this preserves every child span, so custom child-span
    attributes such as ``rag.retrieve.top_score`` on the ``retrieve`` span
    are visible. This is the export-path evidence for environments without
    browser access to the Phoenix UI.
    """
    if df is None or len(df) == 0:
        return "[]"

    recent_traces = (
        df.sort_values("start_time", ascending=False)
        .drop_duplicates("context.trace_id")["context.trace_id"]
        .head(last_n)
        .tolist()
    )
    rows = df[df["context.trace_id"].isin(recent_traces)].sort_values("start_time")

    records = []
    for _, row in rows.iterrows():
        rag_attrs = row.get("attributes.rag")
        records.append(
            {
                "trace_id": str(row.get("context.trace_id", ""))[:8],
                "span": str(row.get("name", "")),
                "rag": rag_attrs if isinstance(rag_attrs, dict) else {},
            }
        )
    return json.dumps(records, indent=2, default=str)


__all__ = [
    "init_tracing",
    "flush",
    "traced_pipeline",
    "summarize_traces",
    "render_markdown",
    "render_json",
    "render_spans_json",
]
