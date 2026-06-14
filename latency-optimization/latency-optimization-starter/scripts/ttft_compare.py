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

# TODO(m26-ex2): time blocking + streaming endpoints on the same question and
# print {"ttft_ms", "total_ms"} for each. See INSTRUCTIONS.md → Exercise 2.
raise NotImplementedError("TODO(m26-ex2): see INSTRUCTIONS.md Exercise 2")
