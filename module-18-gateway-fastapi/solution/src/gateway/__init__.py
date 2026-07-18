"""FastAPI gateway for the ScikitDocs starter.

Public surface:

- ``app`` — the FastAPI application factory output; ``make serve``
  imports it as ``src.gateway.app:app`` and uvicorn boots it on
  ``constants.SERVICE_PORT`` (8080).
- ``router`` — the API router mounted on ``app`` (``POST /query`` +
  ``GET /health``). Re-exported for tests that want to mount it on a
  scratch app.
- ``route_query`` — the in-process dispatch helper the route handler
  calls (classify → tier-select → cache+trace → log). Re-exported so
  the A/B layer can wrap it with variant selection.

The gateway is the convergence point for the rest of the system: the
pipeline, the tracing, the cost log, and the cache all bolt onto this
one HTTP surface.
"""

from src.gateway.app import app, create_app
from src.gateway.router import route_query, select_model
from src.gateway.routes import router

__all__ = ["app", "create_app", "router", "route_query", "select_model"]
