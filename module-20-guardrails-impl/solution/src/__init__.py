"""ScikitDocs starter — shared infrastructure for Course 2 implementation modules.

The system is organised one file (or package) per capability:
- `generator.py` — prompt rendering + OpenAI generation
- `store.py`, `embedder.py`, `chunker.py` — vector store + embeddings + chunking
- `pipeline.py` — end-to-end RAG composition
- `tracing.py` — Phoenix tracing instrumentation
- `scripts/run_eval.py` — RAGAS evaluation harness
- `cost/`, `pricing.py` — cost tracking + pricing
- `cache/` — semantic cache layer
- `gateway/` — FastAPI app + `X-Client-Id` contract
- `guardrails/` — input + output guardrails
- `optimization/` — A/B testing with sticky-by-user via `client_id`
- `ingestion/` — blue/green index swap for RAGOps
- `streaming.py` — streaming endpoint for latency work

Read `INTERFACES.md` (repo root for this starter) for the frozen
function contracts. Read `CONSTANTS.md` for the 21 invariants.
"""
