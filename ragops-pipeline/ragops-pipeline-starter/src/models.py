"""Shared Pydantic models for the ScikitDocs starter.

Mirrors `project/src/models.py` exactly so a learner who has read the
capstone reads identical type shapes here. Don't add ScikitDocs-specific
fields — they belong in `src/store.py` metadata or `src/pipeline.py`
locals, not in the shared response surface.

`QueryResponseValidator` (M20 exercise 4) is the one local-only addition.
It does not mirror `project/` — it is the curriculum-only structured-output
guard the learner wires at the gateway boundary in Skill Pair 9.
"""

from pydantic import BaseModel, Field


class Source(BaseModel):
    """A retrieved scikit-learn doc chunk with its similarity score."""

    doc_id: str
    chunk_text: str
    similarity_score: float


class TokenUsage(BaseModel):
    """Token counts for a single LLM call."""

    prompt_tokens: int
    completion_tokens: int

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class QueryResponse(BaseModel):
    """Standardised response returned by the ScikitDocs `/query` route."""

    answer: str
    sources: list[Source]
    confidence: float
    model: str
    tokens: TokenUsage
    cost_usd: float
    cached: bool = False
    trace_id: str | None = None
    blocked_by: str | None = None


class QueryResponseValidator(BaseModel):
    """Structured-output guard for the `/query` boundary (Skill Pair 9, ex. 4).

    The planning doc names this "the QueryResponse validator with answer +
    citations ≥1 + confidence ∈ [0,1]." Our `QueryResponse` already exists
    and uses `sources` (mirrors `project/`), so this companion model adds
    the missing constraints without renaming the field downstream code
    depends on. `sources` is the same concept the planning doc calls
    `citations` — at the gateway boundary the dict shape is identical.

    What this enforces that `QueryResponse` alone does not:

    - `sources` must be non-empty (`min_length=1`). The hallucination class
      we care about is "answer that cites nothing"; the LLM-judge catches
      *wrong* citations, this validator catches *missing* ones.
    - `confidence` must be in `[0.0, 1.0]`. The base model accepts any
      float; the validator pins the contract a downstream consumer can
      trust without re-checking.

    Used by the boundary block in `src/gateway/routes.py` that wraps the
    dispatched response in `QueryResponseValidator.model_validate(...)`
    and returns HTTP 502 on `ValidationError`.
    """

    answer: str
    sources: list[Source] = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
