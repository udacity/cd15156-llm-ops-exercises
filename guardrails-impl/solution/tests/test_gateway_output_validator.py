"""Tests for the gateway boundary output validator (Module 20 exercise 4).

Two tests pin the contract Skill Pair 9 exercise 4 teaches:

1. Well-formed response (mock ``route_query`` to return a `QueryResponse`
   that satisfies the ``citations`` ≥1 + ``confidence`` ∈ [0, 1]
   constraints) → endpoint returns 200 and the body round-trips through
   ``QueryResponse``.
2. Citation-stripped response (mock ``route_query`` to return a dict
   with ``citations=[]``) → endpoint returns 502 with a body naming
   ``citations`` as the failing field.

The hallucination check is bypassed via a direct patch so the validator
seam is what each test is actually exercising — these tests are about
the structured-output guard, not the LLM-judge.
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from src import constants
from src.guardrails import reset_rate_limit_state
from src.models import QueryResponse, Source, TokenUsage


def _well_formed_response() -> QueryResponse:
    """A `QueryResponse` that satisfies the boundary validator constraints."""
    return QueryResponse(
        answer="StandardScaler standardizes features by removing the mean.",
        citations=[
            Source(
                doc_id="preprocessing/StandardScaler",
                chunk_text="StandardScaler centers and scales features.",
                similarity_score=0.92,
            )
        ],
        confidence=0.92,
        model=constants.MODEL_SIMPLE,
        tokens=TokenUsage(prompt_tokens=80, completion_tokens=20),
        cost_usd=0.0001,
    )


def _citation_stripped_response() -> QueryResponse:
    """A `QueryResponse` with no citations — fails the validator's `min_length=1`.

    We use ``model_construct`` to bypass the constructor's own
    validation (the stricter ``QueryResponse`` rejects ``citations=[]``
    on construction); the test exercises the *boundary re-validation*
    of an already-built response.
    """
    return QueryResponse.model_construct(
        answer="StandardScaler standardizes features by removing the mean.",
        citations=[],  # the validation failure
        confidence=0.5,
        model=constants.MODEL_SIMPLE,
        tokens=TokenUsage(prompt_tokens=80, completion_tokens=20),
        cost_usd=0.0001,
    )


def test_query_endpoint_passes_well_formed_response() -> None:
    """A well-formed response passes the validator and returns 200."""
    reset_rate_limit_state()
    from src.gateway.app import app

    with (
        patch(
            "src.gateway.routes.route_query",
            return_value=_well_formed_response(),
        ),
        patch(
            "src.gateway.routes.check_hallucination",
            return_value=(True, None),
        ),
    ):
        client = TestClient(app)
        response = client.post(
            constants.QUERY_ROUTE,
            json={"question": "What is StandardScaler?"},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"].startswith("StandardScaler")
    assert len(body["citations"]) == 1
    assert 0.0 <= body["confidence"] <= 1.0


def test_query_endpoint_returns_502_on_citation_stripped_response() -> None:
    """A response with empty `citations` fails the validator and returns 502."""
    reset_rate_limit_state()
    from src.gateway.app import app

    with (
        patch(
            "src.gateway.routes.route_query",
            return_value=_citation_stripped_response(),
        ),
        patch(
            "src.gateway.routes.check_hallucination",
            return_value=(True, None),
        ),
    ):
        client = TestClient(app)
        response = client.post(
            constants.QUERY_ROUTE,
            json={"question": "What is StandardScaler?"},
        )

    assert response.status_code == 502, response.text
    body = response.json()
    assert body["detail"] == "output_validation_failed"
    assert body["field"] == "citations"
