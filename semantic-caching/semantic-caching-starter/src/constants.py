"""Authoritative cross-module invariants for the ScikitDocs starter.

Every Course 2 implementation module that touches port numbers, model
names, top-k values, route paths, or any other shared value imports it
from this file. Hardcoding any of these values inside a module is a
review-blocker — the `make consistency-check` target (wired up in
REQ-078) greps for the bare literals.

The set was frozen during REQ-061 ("Scaffold module-starters/scikit-docs/").
Changing a value here counts as an infra amendment and must follow the
infra-amendment protocol in
`docs/plans/2026-05-17-feat-rewrite-impl-modules-scikitdocs-altworkload-plan.md`
§"Operational protocols": branch a REQ-061a patch, re-verify already-shipped
impl REQs' code-refs, document the change.

Twenty-one invariants, grouped by concern.
"""

# === Service ===
SERVICE_PORT: int = 8080
QUERY_ROUTE: str = "/query"
HEALTH_ROUTE: str = "/health"

# === Models — match capstone `project/src/pricing.py` ===
MODEL_COMPLEX: str = "gpt-4o"
MODEL_SIMPLE: str = "gpt-4o-mini"
EMBEDDING_MODEL: str = "text-embedding-3-small"
EMBEDDING_DIM: int = 1536

# === RAG defaults ===
DEFAULT_TOP_K: int = 5
CHUNK_TARGET_TOKENS: int = 500
CHUNK_OVERLAP_TOKENS: int = 75
CONFIDENCE_THRESHOLD: float = 0.7
GENERATION_TEMPERATURE: float = 0.2

# === Evaluation (RAGAS) — REQ-063 / REQ-068 (M11) ===
GOLDEN_SET_SIZE: int = 30
JUDGE_TEMPERATURE: float = 0.0

# === Caching — REQ-070 (M15) ===
CACHE_SIMILARITY_THRESHOLD: float = 0.85

# === Cost logging — REQ-069 (M13) ===
COST_LOG_PATH: str = "data/cost_log.jsonl"

# === Tracing — REQ-067 (M09) ===
PHOENIX_PORT: int = 6006
PHOENIX_PROJECT_NAME: str = "scikitdocs"

# === Gateway + A/B routing — REQ-071 (M18) / REQ-073 (M22) ===
CLIENT_ID_HEADER: str = "X-Client-Id"

# === OpenAI / Vocareum bridge — matches memory `project_vocareum_deployment.md` ===
OPENAI_BASE_URL_ENV: str = "OPENAI_BASE_URL"
VOCAREUM_BASE_URL: str = "https://openai.vocareum.com/v1"


# Exported for `tests/test_smoke.py` — every name listed here must be
# defined above. If you add an invariant, add the name here too.
LOCKED_INVARIANTS: tuple[str, ...] = (
    "SERVICE_PORT",
    "QUERY_ROUTE",
    "HEALTH_ROUTE",
    "MODEL_COMPLEX",
    "MODEL_SIMPLE",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "DEFAULT_TOP_K",
    "CHUNK_TARGET_TOKENS",
    "CHUNK_OVERLAP_TOKENS",
    "CONFIDENCE_THRESHOLD",
    "GENERATION_TEMPERATURE",
    "GOLDEN_SET_SIZE",
    "JUDGE_TEMPERATURE",
    "CACHE_SIMILARITY_THRESHOLD",
    "COST_LOG_PATH",
    "PHOENIX_PORT",
    "PHOENIX_PROJECT_NAME",
    "CLIENT_ID_HEADER",
    "OPENAI_BASE_URL_ENV",
    "VOCAREUM_BASE_URL",
)
