"""FastAPI application factory for the ScikitDocs gateway (REQ-071, M18).

``make serve`` boots ``uvicorn src.gateway.app:app --port 8080``, which
imports ``app`` from this module. ``app`` is the output of
:func:`create_app`, a thin factory that mounts three routers:

- :data:`src.gateway.routes.router` — ``POST /query`` + ``GET /health``
- :data:`src.cost.dashboard.router` — ``GET /cost-dashboard`` (M13)
- :data:`src.streaming.streaming_router` — ``POST /query/stream`` (M26)

The streaming router is a curriculum-later addition (M26) mounted on a
curriculum-earlier surface (M18). The app is a wiring layer and is
allowed to know about every package; importing the streaming router
here is the documented exception to the forward-dependency rule the
gateway otherwise honors.

The :func:`lifespan` async context manager boots Phoenix tracing on
startup (M09) and force-flushes any in-flight spans on shutdown so the
last few requests are not silently dropped. The two side effects mirror
the capstone's ``project/src/gateway/app.py`` exactly — same hooks,
same ordering.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.cost.dashboard import router as cost_router
from src.gateway.routes import router as api_router
from src.streaming import streaming_router
from src.tracing import flush, init_tracing


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Boot tracing on startup; flush queued spans on shutdown.

    ``init_tracing`` is idempotent and respects
    ``settings.tracing_backend == "none"`` as a kill-switch, so the
    test suite can mount this app without a Phoenix dependency.
    """
    init_tracing()
    try:
        yield
    finally:
        flush()


def create_app() -> FastAPI:
    """Construct the FastAPI app with both routers mounted.

    Kept as a factory (rather than a module-level construction) so
    tests can build a fresh app per test when needed — the default
    ``app`` instance below covers the production path.
    """
    application = FastAPI(title="ScikitDocs Gateway", lifespan=lifespan)
    application.include_router(api_router)
    application.include_router(cost_router)
    application.include_router(streaming_router)
    return application


app = create_app()
