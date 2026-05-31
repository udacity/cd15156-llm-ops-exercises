"""Exercise 3 — sweep `ef_search` against parallel sandbox collections.

Pulls every chunk out of the live `scikit_docs` collection, copies them into
three parallel collections built with `ef_search` of 10, 50, and 200, then
runs the same five queries twenty times against each collection and reports
mean per-query latency. Embedding calls are done up front and excluded from
the timing so the measurement is the vector search alone.

Does not touch `src/store.py` or the live `scikit_docs` collection. The
parallel collections are dropped between iterations and can be cleaned up
afterward with the snippet at the bottom of Exercise 3.

Run with `make load-data` already complete:

    uv run python scripts/ef_search_sweep.py
"""

# TODO(m26-ex3): build sandbox collections at ef_search of 10/50/200, replay the
# same queries against each, and print mean per-query latency. See
# INSTRUCTIONS.md → Exercise 3 for the query list, sandbox-collection pattern,
# and cleanup snippet.
raise NotImplementedError("TODO(m26-ex3): see INSTRUCTIONS.md Exercise 3")
