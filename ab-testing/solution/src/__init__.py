"""ScikitDocs starter — shared infrastructure for Course 2 implementation modules.

Each impl module fills in the file it teaches:
- M03 (REQ-064) → `generator.py`
- M05 (REQ-065) → `store.py`, `embedder.py`, `chunker.py`
- M07 (REQ-066) → `pipeline.py`
- M09 (REQ-067) → adds tracing instrumentation across the stack
- M11 (REQ-068) → `scripts/run_eval.py`
- M13 (REQ-069) → cost tracking + pricing
- M15 (REQ-070) → semantic cache layer
- M18 (REQ-071) → `gateway/` package with FastAPI app + `X-Client-Id` contract
- M20 (REQ-072) → guardrails (input + output)
- M22 (REQ-073) → A/B testing with sticky-by-user via `client_id`
- M24 (REQ-074) → blue/green index swap for RAGOps
- M26 (REQ-075) → streaming endpoint for latency demo

Read `INTERFACES.md` (repo root for this starter) for the frozen
function contracts. Read `CONSTANTS.md` for the 21 invariants.
"""
