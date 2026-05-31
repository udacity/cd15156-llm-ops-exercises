"""FastAPI router exposing ``GET /cost-dashboard`` as a simple HTML report (REQ-069, M13).

Two ways to serve this:

1. **As part of the full gateway (REQ-071 / M18 wires it in):** the M18
   gateway mounts ``router`` on the main FastAPI app at startup, so the
   dashboard lives alongside ``/query`` and ``/health`` on port 8080.

2. **Standalone (this module's ``__main__``):** ``uv run python -m
   src.cost.dashboard`` spins a tiny FastAPI app on
   ``constants.SERVICE_PORT`` (8080) with just the dashboard mounted.
   That's what ``make cost-dashboard`` uses so M13 learners can render
   the dashboard without waiting for M18.

Every value interpolated into the HTML is run through ``html.escape``
even though the cost log is currently operator-controlled. The escape
calls are defense in depth: if a future version filters by query type
or model name from a request parameter, the existing template stays
safe.
"""

import html

from fastapi import APIRouter, FastAPI
from fastapi.responses import HTMLResponse

from src import constants
from src.cost.tracker import load_log, summarize

router = APIRouter()


def _row(label: str, value: str) -> str:
    """Render one ``<tr>`` for the totals table, escaping both columns."""
    return f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"


def render_html(summary: dict) -> str:
    """Render the summary dict as a small standalone HTML page."""
    by_model_rows = "".join(
        f"<tr>"
        f"<td>{html.escape(model)}</td>"
        f"<td>{stats['requests']}</td>"
        f"<td>${stats['cost_usd']:.4f}</td>"
        f"<td>${stats['avg_cost_usd']:.6f}</td>"
        f"</tr>"
        for model, stats in sorted(summary["by_model"].items())
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Cost Dashboard</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
    table {{ border-collapse: collapse; margin-top: 1rem; }}
    th, td {{ padding: 0.4rem 0.8rem; border: 1px solid #ddd; text-align: left; }}
    th {{ background: #f5f5f5; }}
    h1 {{ margin-bottom: 0; }}
  </style>
</head>
<body>
  <h1>Cost Dashboard</h1>
  <p>Aggregate cost data from the local JSONL request log.</p>
  <table>
    {_row("Total requests", str(summary["total_requests"]))}
    {_row("Total cost (USD)", f"${summary['total_cost_usd']:.4f}")}
  </table>
  <h2>Per-model breakdown</h2>
  <table>
    <thead>
      <tr><th>Model</th><th>Requests</th><th>Cost (USD)</th><th>Avg cost / request</th></tr>
    </thead>
    <tbody>
      {by_model_rows or '<tr><td colspan="4">No requests logged yet.</td></tr>'}
    </tbody>
  </table>
</body>
</html>
"""


@router.get("/cost-dashboard", response_class=HTMLResponse)
async def cost_dashboard() -> str:
    """Serve a small HTML report of total + per-model cost.

    Reads the JSONL log at ``settings.cost_log_path`` on every request.
    No caching, no pagination — the log is operator-scale (one append
    per HTTP query). For historical trending in a production deployment,
    either query the log directly or replace this endpoint with a
    Grafana/Prometheus integration.
    """
    return render_html(summarize(load_log()))


# Standalone app — used by ``make cost-dashboard`` / ``uv run python -m
# src.cost.dashboard``. REQ-071 (M18) mounts ``router`` on the full
# gateway app instead, at which point this standalone path is just one
# extra way to reach the same view.
app = FastAPI(title="ScikitDocs Cost Dashboard (standalone)")
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=constants.SERVICE_PORT)
