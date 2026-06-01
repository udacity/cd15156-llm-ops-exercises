"""Shared Pydantic models for the ScikitDocs starter.

Mirrors `project/src/models.py` for the most part. The local-only
deviation (Module 20 exercise 4) is that `QueryResponse.citations` and
`QueryResponse.confidence` carry Pydantic `Field` constraints — the
structured-output contract the learner wires at the gateway boundary
in Skill Pair 9.
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
    # Structured-output contract: citations require ≥1 source, confidence in [0.0, 1.0]
    citations: list[Source] = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    model: str
    tokens: TokenUsage
    cost_usd: float
    cached: bool = False
    trace_id: str | None = None
    blocked_by: str | None = None
