"""Prompt-optimization primitives for the ScikitDocs starter (Module 22).

Module 22 teaches A/B testing as the discipline that operationalizes
Module 02's *attribution* property — routing some fraction of live
traffic to a prompt variant and measuring whether the change moved a
metric you defined in advance. The package ships the production-grade
routing primitive (`pick_variant`), the variant-aware OpenAI caller
(`call_with_variant`), and the per-call JSONL logger
(`log_assignment`).

The cross-module contract is the `X-Client-Id` header added to
`src/gateway/routes.py`. When a request arrives with the header,
`src.gateway.router.route_query` forwards the value as `client_id`;
Module 22's `pick_variant(client_id, traffic_split)` hashes it through
SHA-256 mod the cumulative-weight bucket to produce a deterministic
sticky-by-user variant assignment. When the header is absent — the
Module 11 RAGAS eval harness fires un-headered requests, for instance —
`pick_variant` falls back to per-request weighted sampling so the
caller still gets a routable answer.

See `INTERFACES.md` (starter root) for the frozen contract and the
Module 21 concept-script (V2 + V3) for the conceptual framing.
"""

from src.optimization.ab import (
    call_with_variant,
    log_assignment,
    pick_variant,
)

__all__ = [
    "pick_variant",
    "call_with_variant",
    "log_assignment",
]
