"""Export recent Phoenix traces as markdown or JSON (REQ-067, M09).

Backstop for environments where the Phoenix UI on port 6006 isn't
reachable from the learner's browser. Reads spans from the embedded
Phoenix server (running because something in this process called
``init_tracing``) and renders a per-trace summary using
``src.tracing.{summarize_traces,render_markdown,render_json}``.

Usage:
    uv run python scripts/show_traces.py                       # markdown
    uv run python scripts/show_traces.py --last 20             # last 20
    uv run python scripts/show_traces.py --json                # JSON
    uv run python scripts/show_traces.py --output traces.md    # write file

Or via the Makefile shortcut:
    make show-traces

Note: Phoenix's embedded store is per-process. If you fired your
queries in one Python invocation and then run this script as a
separate one, the spans are gone. Use ``scripts/seed_traces.py`` for
the in-process fire-and-export pattern that produces the rubric §7
evidence file at ``data/trace_evidence.md``.
"""

import argparse
import sys
import warnings
from pathlib import Path

from src.config import settings
from src.tracing import render_json, render_markdown, summarize_traces


def _fetch_spans():
    """Pull all spans from the in-process Phoenix server.

    Spans land in ``settings.phoenix_project_name`` (the tracer
    provider is registered with that project), so ``phoenix.Client``
    must pass it through. Without it, the client queries the
    ``default`` project and returns an empty dataframe.

    Stays on ``phoenix.Client`` (GraphQL) rather than the newer
    ``arize-phoenix-client`` REST client — the embedded Phoenix's
    REST endpoint doesn't expose in-memory spans the same way, so
    migrating yields empty dataframes. The deprecation warning the
    legacy client emits is filtered here.
    """
    import phoenix

    # ``phoenix_host`` is the bind address (often ``0.0.0.0``). Clients
    # need a routable address — fall back to ``127.0.0.1`` when the
    # bind is the wildcard.
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
        description="Export recent Phoenix traces as markdown or JSON."
    )
    parser.add_argument(
        "--last",
        type=int,
        default=10,
        help="Number of most-recent traces to include (default: 10).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of markdown.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Write the output to a file instead of stdout.",
    )
    args = parser.parse_args(argv)

    try:
        df = _fetch_spans()
    except Exception as exc:
        print(
            f"Could not connect to Phoenix at "
            f"http://{settings.phoenix_host}:{settings.phoenix_port}.",
            file=sys.stderr,
        )
        print(
            f"Is a process running that called `init_tracing()`? "
            f"Error: {exc}",
            file=sys.stderr,
        )
        return 1

    summaries = summarize_traces(df, last_n=args.last)
    total = 0 if df is None else len(df.groupby("context.trace_id"))
    output = (
        render_json(summaries) if args.json else render_markdown(summaries, total)
    )

    if args.output:
        Path(args.output).write_text(output)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
