"""Scaffold-level smoke test for the ScikitDocs starter.

Runs at scaffolding. Verifies the shape of
the starter without invoking any of the stub function bodies — stubs
will raise NotImplementedError, which is the correct behavior until
the per-module REQs fill them in.

A real end-to-end smoke test (loaded corpus + answered query) is added
by once the infra REQs land. The ``X-Client-Id`` header contract
test (``test_x_client_id_*``) is added at scaffolding and pins the
sticky-by-user contract Module 22 builds on.
"""

import importlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src import constants
from src.models import QueryResponse, Source, TokenUsage


STARTER_ROOT = Path(__file__).resolve().parents[1]

STUBBED_SRC_MODULES = ()

# Modules that started as stubs and have since been filled.
FILLED_SRC_MODULES = (
    "src.corpus",
    "src.generator",  # Module 03
    "src.chunker",  # Module 05
    "src.embedder",  # Module 05
    "src.store",  # Module 05
    "src.pipeline",  # Module 07
)

REAL_SRC_MODULES = (
    "src",
    "src.constants",
    "src.models",
    "src.config",
    "src.pricing",  # Module 13
    "src.cost",  # Module 13
    "src.cost.tracker",  # Module 13
    "src.cost.dashboard",  # Module 13
    "src.cache",  # Module 15
    "src.cache.semantic",  # Module 15
    "src.cache.wrapper",  # Module 15
    "src.evaluation",  # Module 11
    "src.evaluation.run_eval",  # Module 11
    "src.evaluation.deprecated_apis",  # Module 11
    "src.gateway",  # Module 18
    "src.gateway.app",  # Module 18
    "src.gateway.classifier",  # Module 18
    "src.gateway.router",  # Module 18
    "src.gateway.routes",  # Module 18
    "src.guardrails",  # Module 20
    "src.guardrails.input_guards",  # Module 20
    "src.guardrails.rate_limit",  # Module 20
    "src.guardrails.wrapper",  # Module 20
    "src.guardrails.llm_judge",  # Module 20
    "src.guardrails.llm_judge.output_guards",  # Module 20
)


@pytest.mark.parametrize(
    "module_name", REAL_SRC_MODULES + STUBBED_SRC_MODULES + FILLED_SRC_MODULES
)
def test_module_imports(module_name: str) -> None:
    """Every src/ module is importable. (Stubs import; bodies raise on call.)"""
    importlib.import_module(module_name)


def test_all_locked_invariants_are_defined() -> None:
    """The 21 invariants enumerated in constants.LOCKED_INVARIANTS are all bound."""
    for name in constants.LOCKED_INVARIANTS:
        assert hasattr(constants, name), f"constants.{name} missing"


def test_locked_invariants_count_is_21() -> None:
    """The lock list is exactly 21 invariants — guards against accidental drift."""
    assert len(constants.LOCKED_INVARIANTS) == 21


def test_critical_invariant_values_match_capstone() -> None:
    """Pin the values that downstream modules depend on. Drift = consistency-check fail."""
    assert constants.SERVICE_PORT == 8080
    assert constants.MODEL_COMPLEX == "gpt-4o"
    assert constants.MODEL_SIMPLE == "gpt-4o-mini"
    assert constants.EMBEDDING_MODEL == "text-embedding-3-small"
    assert constants.EMBEDDING_DIM == 1536
    assert constants.DEFAULT_TOP_K == 5
    assert constants.PHOENIX_PORT == 6006
    assert constants.PHOENIX_PROJECT_NAME == "scikitdocs"
    assert constants.QUERY_ROUTE == "/query"
    assert constants.HEALTH_ROUTE == "/health"
    assert constants.CLIENT_ID_HEADER == "X-Client-Id"
    assert constants.OPENAI_BASE_URL_ENV == "OPENAI_BASE_URL"


def test_stub_function_signatures_present() -> None:
    """Each stub module exposes the functions documented in INTERFACES.md."""
    expected = {
        "src.corpus": ("load_corpus",),
        "src.chunker": ("chunk_doc",),
        "src.embedder": ("embed", "embed_query"),
        "src.store": ("get_collection", "add", "query"),
        "src.generator": ("render_system_prompt", "generate"),
        "src.pipeline": ("run_pipeline",),
    }
    for module_name, fn_names in expected.items():
        module = importlib.import_module(module_name)
        for fn_name in fn_names:
            assert callable(getattr(module, fn_name, None)), (
                f"{module_name}.{fn_name} missing or not callable"
            )


# ``test_stubs_raise_notimplemented`` lived here until every entry in
# STUBBED_SRC_MODULES was scaffolded/064/065/066. Per-module
# behavior is now covered by dedicated tests (see ``test_corpus.py`` for
# the original example pattern). The narrowing protocol carries forward:
# when a new stub lands and is later filled, remove the corresponding
# ``pytest.raises`` block and move the module into FILLED_SRC_MODULES.


def test_starter_files_present() -> None:
    """Required top-level files exist at the starter root."""
    expected = (
        "README.md",
        "INTERFACES.md",
        "CONSTANTS.md",
        "Makefile",
        "pyproject.toml",
        ".env.example",
        "prompts/docbot_system.j2",
        "prompts/classifier.j2",  # Module 18
    )
    for relative in expected:
        assert (STARTER_ROOT / relative).exists(), f"{relative} missing"


# === X-Client-Id contract test ===
#
# Pins the cross-module contract Module 22 depends on: the
# ``X-Client-Id`` request header threads through ``POST /query`` and
# arrives at :func:`src.gateway.router.route_query` as the ``client_id``
# keyword argument. Module 22 will read it for sticky-by-user variant
# assignment. Module 18 itself does nothing with the value beyond forwarding
# it — the test below pins the plumbing, not the consumer.


def _stub_response() -> QueryResponse:
    """Cheap ``QueryResponse`` for the contract tests — no LLM call needed."""
    return QueryResponse(
        answer="stub",
        sources=[Source(doc_id="doc_1", chunk_text="stub", similarity_score=0.9)],
        confidence=0.9,
        model=constants.MODEL_SIMPLE,
        tokens=TokenUsage(prompt_tokens=10, completion_tokens=5),
        cost_usd=0.0,
        cached=False,
    )


def test_x_client_id_header_passes_through_to_router() -> None:
    """Header sent → ``route_query`` receives the value as ``client_id``."""
    from src.gateway.app import app

    captured: dict[str, str | None] = {}

    def fake_route_query(
        question: str, top_k: int = 5, *, model: str | None = None, client_id: str | None = None
    ) -> QueryResponse:
        captured["client_id"] = client_id
        return _stub_response()

    with patch("src.gateway.routes.route_query", side_effect=fake_route_query):
        client = TestClient(app)
        response = client.post(
            constants.QUERY_ROUTE,
            json={"question": "default criterion for RandomForestRegressor?"},
            headers={constants.CLIENT_ID_HEADER: "user-42"},
        )

    assert response.status_code == 200, response.text
    assert captured["client_id"] == "user-42"


def test_x_client_id_header_is_optional() -> None:
    """No header sent → ``route_query`` receives ``client_id=None`` and 200 OK."""
    from src.gateway.app import app

    captured: dict[str, str | None] = {}

    def fake_route_query(
        question: str, top_k: int = 5, *, model: str | None = None, client_id: str | None = None
    ) -> QueryResponse:
        captured["client_id"] = client_id
        return _stub_response()

    with patch("src.gateway.routes.route_query", side_effect=fake_route_query):
        client = TestClient(app)
        response = client.post(
            constants.QUERY_ROUTE,
            json={"question": "What is StandardScaler?"},
        )

    assert response.status_code == 200, response.text
    assert captured["client_id"] is None


def test_health_route_serves_200() -> None:
    """``GET /health`` is a static liveness probe — no dependencies, no I/O."""
    from src.gateway.app import app

    client = TestClient(app)
    response = client.get(constants.HEALTH_ROUTE)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# === POST /query/stream SSE contract tests ===
#
# Pins three properties of the streaming endpoint Module 26 builds against:
# the SSE media type, at least one ``token`` event, and exactly one
# trailing ``done`` event whose ``response`` payload parses as a
# ``QueryResponse``. OpenAI + Chroma + the embedder are all patched so
# the tests run offline at zero LLM spend.


def _fake_stream(question, sources, model):
    """Stand-in for :func:`src.streaming.stream_completion`."""
    yield "hello"
    yield " "
    yield "world"
    yield (
        "hello world",
        TokenUsage(prompt_tokens=12, completion_tokens=2),
        0.0001,
    )


def test_stream_endpoint_returns_sse_media_type() -> None:
    """``POST /query/stream`` returns 200 + ``text/event-stream`` content type."""
    from src.gateway.app import app

    with (
        patch("src.streaming.embed_query", return_value=[0.0] * 8),
        patch("src.streaming.store_query", return_value=[]),
        patch("src.gateway.classifier.classify", return_value="simple"),
        patch("src.gateway.router.select_model", return_value=constants.MODEL_SIMPLE),
        patch("src.streaming.stream_completion", side_effect=_fake_stream),
        patch("src.streaming.log_request", return_value=None),
    ):
        client = TestClient(app)
        response = client.post(
            "/query/stream",
            json={"question": "What is StandardScaler?"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


def test_stream_endpoint_yields_done_event() -> None:
    """The streamed body ends with a ``done`` event whose payload parses as ``QueryResponse``."""
    from src.gateway.app import app

    with (
        patch("src.streaming.embed_query", return_value=[0.0] * 8),
        patch("src.streaming.store_query", return_value=[]),
        patch("src.gateway.classifier.classify", return_value="simple"),
        patch("src.gateway.router.select_model", return_value=constants.MODEL_SIMPLE),
        patch("src.streaming.stream_completion", side_effect=_fake_stream),
        patch("src.streaming.log_request", return_value=None),
    ):
        client = TestClient(app)
        response = client.post(
            "/query/stream",
            json={"question": "How do I tune n_estimators?"},
        )

    events = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events, "expected at least one SSE data frame"

    final = json.loads(events[-1])
    assert final["type"] == "done"
    QueryResponse.model_validate(final["response"])


def test_stream_endpoint_streams_token_events() -> None:
    """At least one ``token`` event is yielded before the ``done`` event."""
    from src.gateway.app import app

    with (
        patch("src.streaming.embed_query", return_value=[0.0] * 8),
        patch("src.streaming.store_query", return_value=[]),
        patch("src.gateway.classifier.classify", return_value="simple"),
        patch("src.gateway.router.select_model", return_value=constants.MODEL_SIMPLE),
        patch("src.streaming.stream_completion", side_effect=_fake_stream),
        patch("src.streaming.log_request", return_value=None),
    ):
        client = TestClient(app)
        response = client.post(
            "/query/stream",
            json={"question": "Pipeline vs ColumnTransformer?"},
        )

    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    token_events = [event for event in events if event.get("type") == "token"]
    done_events = [event for event in events if event.get("type") == "done"]

    assert len(token_events) >= 1, "expected at least one token frame before done"
    assert len(done_events) == 1, "expected exactly one done frame"
    assert events[-1]["type"] == "done"
