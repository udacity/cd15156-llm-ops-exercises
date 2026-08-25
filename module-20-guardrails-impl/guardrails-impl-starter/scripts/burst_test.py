# TODO(m20-exercise-3): burst demonstration — fire 25 sequential /query requests
# from one client id and show the LLM10 token bucket block requests 21-25.
# Complete this scaffold. The finished reference lives in solution/ under the same
# marker. The max_tokens plumbing for Steps 1-2 is wired at the other
# TODO(m20-exercise-3) markers in src/generator.py, src/pipeline.py,
# src/tracing.py, and src/gateway/router.py.
"""Demonstrate the LLM10 unbounded-consumption block under a burst.

Fires ``--count`` sequential POSTs at ``/query`` under one ``X-Client-Id``. With
the shipped defaults (``RATE_LIMIT_REQUESTS = 20`` per 60s), the first twenty
return a normal answer and the twenty-first onward return HTTP 200 with
``blocked_by`` set to the ``unbounded_consumption`` reason. The bucket is
process-local, so it resets on ``make serve`` restart.

Needs ``make serve`` up on :8080 and a loaded corpus; each of the first twenty
calls is a real model round-trip. Run one throwaway query first so the DeBERTa
cold-start (~700 MB) is not counted in the burst.

Usage:
    PYTHONPATH=. uv run python scripts/burst_test.py
    PYTHONPATH=. uv run python scripts/burst_test.py --count 25 --client-id burst-test
"""

import argparse
import json
import urllib.error
import urllib.request

URL = "http://localhost:8080/query"


def fire(question: str, client_id: str) -> dict:
    payload = json.dumps({"question": question}).encode()
    req = urllib.request.Request(
        URL,
        data=payload,
        headers={"content-type": "application/json", "X-Client-Id": client_id},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--client-id", default="burst-test")
    parser.add_argument("--question", default="What is StandardScaler?")
    args = parser.parse_args()

    # Exercise 3, Step 3: fire args.count sequential requests under one client id
    # via fire(). Print a per-request status and flag the first request whose
    # response carries "unbounded_consumption" in blocked_by. The first 20 pass;
    # 21-25 block. See INSTRUCTIONS.md Exercise 3.
    raise NotImplementedError("Complete main — see INSTRUCTIONS.md Exercise 3")


if __name__ == "__main__":
    main()
