"""In-process semantic cache for the ScikitDocs starter.

Public surface:

- ``lookup`` — embed a question, similarity-search the cache collection,
  apply the cosine-distance threshold gate, return a cached
  ``QueryResponse`` (with ``cached=True``) on hit or ``None`` on miss.
- ``store`` — embed a question and upsert a cached response with a UUID
  key and four metadata fields.
- ``clear`` — drop every entry in the cache collection.
- ``cached_route_query`` — convenience composition of
  ``lookup → run_pipeline → store`` used by the demo and exercises
  in place of an HTTP route.
"""

from src.cache.semantic import COLLECTION_NAME, clear, lookup, store
from src.cache.wrapper import cached_route_query

__all__ = ["lookup", "store", "clear", "cached_route_query", "COLLECTION_NAME"]
