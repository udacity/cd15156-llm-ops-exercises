"""Exercise 2 — TTFT comparison: blocking /query vs streaming /query/stream.

Times both endpoints on the same question. Blocking TTFT equals total
(no body until the whole response lands). Streaming TTFT lands much lower
because the first SSE frame arrives as soon as the model starts generating.
The two totals land close together: the model does the same work either way,
so streaming does not make generation faster, it just surfaces the first token
sooner. (With the output guard enabled, blocking would carry an
extra hallucination-judge LLM call after the last token that the streaming
route defers; this module ships with that guard off so the totals compare
cleanly.)

Run with `make serve` up on port 8080 and the cache cleared once before the
run (streaming bypasses the cache anyway; the clear is only what makes the
blocking call a cold miss):

    uv run python -c "from src.cache import clear; clear()"
    uv run python scripts/ttft_compare.py
"""

# TODO(m26-ex2): time blocking + streaming endpoints on the same question and
# print {"ttft_ms", "total_ms"} for each. See INSTRUCTIONS.md → Exercise 2.
import json
import time
import urllib.request

QUESTION = "How do I serialize a fitted scikit-learn Pipeline?"


def time_blocking() -> dict:
    req = urllib.request.Request(
        "http://localhost:8080/query",
        data=json.dumps({"question": QUESTION}).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        resp.read()
    total_ms = round((time.perf_counter() - start) * 1000)
    return {"ttft_ms": total_ms, "total_ms": total_ms}


def time_streaming() -> dict:
    req = urllib.request.Request(
        "http://localhost:8080/query/stream",
        data=json.dumps({"question": QUESTION}).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    ttft_ms = None
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            if ttft_ms is None:
                ttft_ms = round((time.perf_counter() - start) * 1000)
    total_ms = round((time.perf_counter() - start) * 1000)
    return {"ttft_ms": ttft_ms or total_ms, "total_ms": total_ms}


if __name__ == "__main__":
    print("blocking :", time_blocking())
    print("streaming:", time_streaming())
