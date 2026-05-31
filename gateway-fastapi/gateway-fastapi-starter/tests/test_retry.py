"""Exercise 2 — retry behavior of _call_chat_completions.

Two cases pinned: 5xx errors retry and eventually succeed; 4xx errors fail
fast without retry. The production decorator uses ``wait_exponential_jitter
(initial=1, max=8)``; the tests reach into the decorator's ``retry``
state machine to override the wait to near-zero so the suite finishes
in milliseconds rather than seconds.
"""

# TODO(m18-ex2): author the two retry tests — 5xx-then-success retries and 4xx fails fast
raise NotImplementedError("TODO(m18-ex2)")
