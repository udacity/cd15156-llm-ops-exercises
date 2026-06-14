"""Exercise 2 — TTFT comparison: blocking /query vs streaming /query/stream.

Times both endpoints on the same question. Blocking TTFT equals total
(no body until the whole response lands). Streaming TTFT lands in the
few-hundred-millisecond range because the first SSE frame arrives as soon
as the model starts generating; streaming total is comparable to blocking
total because the model still has to finish generating.

Run with `make serve` up on port 8080 and the cache cleared between calls
(streaming bypasses the cache anyway; clear the cache so blocking misses too):

    uv run python -c "from src.cache import clear; clear()"
    uv run python scripts/ttft_compare.py
"""

# Time blocking and streaming endpoints on the same question and print
# {"ttft_ms", "total_ms"} for each.
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
    total_ms = (time.perf_counter() - start) * 1000
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
                ttft_ms = (time.perf_counter() - start) * 1000
    total_ms = (time.perf_counter() - start) * 1000
    return {"ttft_ms": ttft_ms or total_ms, "total_ms": total_ms}


if __name__ == "__main__":
    print("blocking :", time_blocking())
    print("streaming:", time_streaming())
