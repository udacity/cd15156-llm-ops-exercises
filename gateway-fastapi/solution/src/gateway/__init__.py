"""FastAPI gateway for the ScikitDocs starter (REQ-071, M18).

Public surface mirrors ``project/src/gateway/__init__.py``:

- ``app`` — the FastAPI application factory output; ``make serve``
  imports it as ``src.gateway.app:app`` and uvicorn boots it on
  ``constants.SERVICE_PORT`` (8080).
- ``router`` — the API router mounted on ``app`` (``POST /query`` +
  ``GET /health``). Re-exported for tests that want to mount it on a
  scratch app.
- ``route_query`` — the in-process dispatch helper the route handler
  calls (classify → tier-select → cache+trace → log). Re-exported so
  M22 (REQ-073) can wrap it with A/B variant selection.

The gateway is the convergence point for every Wave 1-3 capability
the starter ships: M07's pipeline, M09's tracing, M13's cost log, and
M15's cache all bolt onto this one HTTP surface.
"""

from src.gateway.app import app, create_app
from src.gateway.router import route_query, select_model
from src.gateway.routes import router

__all__ = ["app", "create_app", "router", "route_query", "select_model"]
