"""Exercise 2 — retry behavior of _call_chat_completions.

Two cases pinned: 5xx errors retry and eventually succeed; 4xx errors fail
fast without retry. The production decorator uses ``wait_exponential_jitter
(initial=1, max=8)``; the tests reach into the decorator's ``retry``
state machine to override the wait to near-zero so the suite finishes
in milliseconds rather than seconds.
"""

# TODO(m18-ex2): author the two retry tests — 5xx-then-success retries and 4xx fails fast

from unittest.mock import MagicMock

import httpx
import pytest
from openai import APIStatusError
from tenacity import wait_exponential_jitter

from src.generator import _call_chat_completions

# Speed up the suite — production policy uses initial=1, max=8 which would
# block the test for several seconds across two retries.
_call_chat_completions.retry.wait = wait_exponential_jitter(initial=0.01, max=0.05)


def _make_5xx() -> APIStatusError:
    return APIStatusError(
        "server error",
        response=httpx.Response(status_code=503, request=httpx.Request("POST", "http://test")),
        body=None,
    )


def test_retries_succeed_after_two_5xx():
    """Two 503s then a success — tenacity should retry and the third call should return."""
    success = MagicMock()
    success.choices = [MagicMock()]
    success.choices[0].message.content = "ok"

    call_results = [_make_5xx(), _make_5xx(), success]

    def side_effect(**kwargs):
        outcome = call_results.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    client = MagicMock()
    client.chat.completions.create.side_effect = side_effect

    result = _call_chat_completions(client, "gpt-4o-mini", "system", "question")
    assert result.choices[0].message.content == "ok"
    assert client.chat.completions.create.call_count == 3


def test_does_not_retry_on_400():
    """A 400 is a client error — retrying re-sends the wrong request."""
    err = APIStatusError(
        "bad request",
        response=httpx.Response(status_code=400, request=httpx.Request("POST", "http://test")),
        body=None,
    )
    client = MagicMock()
    client.chat.completions.create.side_effect = err

    with pytest.raises(APIStatusError):
        _call_chat_completions(client, "gpt-4o-mini", "system", "question")
    assert client.chat.completions.create.call_count == 1
