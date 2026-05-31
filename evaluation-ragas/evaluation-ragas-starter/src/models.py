"""Shared Pydantic models for the ScikitDocs starter.

Mirrors `project/src/models.py` exactly so a learner who has read the
capstone reads identical type shapes here. Don't add ScikitDocs-specific
fields — they belong in `src/store.py` metadata or `src/pipeline.py`
locals, not in the shared response surface.
"""

from pydantic import BaseModel


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
