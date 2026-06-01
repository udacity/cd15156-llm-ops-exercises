"""FastAPI gateway for the ScikitDocs starter (Module 18).

Public surface mirrors ``project/src/gateway/__init__.py``:

- ``app`` — the FastAPI application factory output; ``make serve``
  imports it as ``src.gateway.app:app`` and uvicorn boots it on
  ``constants.SERVICE_PORT`` (8080).
- ``router`` — the API router mounted on ``app`` (``POST /query`` +
  ``GET /health``). Re-exported for tests that want to mount it on a
  scratch app.
- ``route_query`` — the in-process dispatch helper the route handler
  calls (classify → tier-select → cache+trace → log). Re-exported so
  Module 22 can wrap it with A/B variant selection.

The gateway is the convergence point for every Wave 1-3 capability
the starter ships: Module 07's pipeline, Module 09's tracing, Module 13's cost log, and
Module 15's cache all bolt onto this one HTTP surface.
"""

from src.gateway.app import app, create_app
from src.gateway.router import route_query, select_model
from src.gateway.routes import router

__all__ = ["app", "create_app", "router", "route_query", "select_model"]
