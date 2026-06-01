"""Seed Phoenix with a 5-question pack and export the rubric §7 evidence (Module 09).

Fires a hand-curated set of ``traced_pipeline`` calls in-process so the
OTel exporter and the markdown renderer both see the same spans, then
writes the per-trace summary to ``data/trace_evidence.md``. That file
is the rubric §7-equivalent artifact for environments where the Phoenix
UI on port 6006 isn't reachable from the learner's browser.

The five questions are deliberately diverse:

- ``factual_in_domain`` — short fact lookup against the scikit-learn docs
- ``version_sensitive`` — value that differs across scikit-learn releases
- ``conceptual_in_domain`` — multi-paragraph explainer (longer completion)
- ``off_topic`` — exercises the refusal path
- ``comparison`` — two-API contrast (more retrieved chunks → larger prompt)

Run via the Makefile shortcut:

    make seed-traces

Or directly:

    uv run python scripts/seed_traces.py
    uv run python scripts/seed_traces.py --output /tmp/traces.md
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

from src.config import settings
from src.tracing import init_tracing, render_markdown, summarize_traces, traced_pipeline

_QUESTIONS = [
    "What is the default value of `n_estimators` in `RandomForestClassifier`?",
    "What solver does `LogisticRegression` use by default in scikit-learn 1.5?",
    "Explain how `StandardScaler` works and when to use it.",
    "What is the weather in Paris today?",
    "Compare `RandomForestClassifier` and `GradientBoostingClassifier` for tabular data.",
]


def _fetch_spans():
    import phoenix

    host = settings.phoenix_host
    if host in ("0.0.0.0", "::", ""):
        host = "127.0.0.1"
    endpoint = f"http://{host}:{settings.phoenix_port}"
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Migrate to .*arize-phoenix-client",
            category=DeprecationWarning,
        )
        client = phoenix.Client(endpoint=endpoint)
        return client.get_spans_dataframe(
            project_name=settings.phoenix_project_name
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Seed Phoenix with traced queries and export the §7 evidence file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/trace_evidence.md",
        help="Where to write the markdown export (default: data/trace_evidence.md).",
    )
    parser.add_argument(
        "--flush-wait-seconds",
        type=float,
        default=3.0,
        help="Seconds to wait after the last query for the OTel exporter to flush.",
    )
    args = parser.parse_args(argv)

    init_tracing()

    print(f"Firing {len(_QUESTIONS)} traced queries...", file=sys.stderr)
    for q in _QUESTIONS:
        response = traced_pipeline(q)
        print(
            f"  trace={response.trace_id[:8] if response.trace_id else '—'} "
            f"model={response.model} "
            f"tokens={response.tokens.total}",
            file=sys.stderr,
        )

    print(
        f"Waiting {args.flush_wait_seconds:.1f}s for OTel exporter to flush...",
        file=sys.stderr,
    )
    time.sleep(args.flush_wait_seconds)

    df = _fetch_spans()
    summaries = summarize_traces(df, last_n=len(_QUESTIONS))
    total = 0 if df is None else len(df.groupby("context.trace_id"))
    output = render_markdown(summaries, total)

    if summaries:
        slowest = max(summaries, key=lambda s: s["latency_ms"])
        output += (
            f"\n**Slowest step across {len(summaries)} traces:** "
            f"`{slowest['slowest_span']}` "
            f"({slowest['slowest_ms']} ms) "
            f"in trace `{slowest['trace_id']}` "
            f"(\"{slowest['question']}\").\n"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output)
    print(f"Wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
