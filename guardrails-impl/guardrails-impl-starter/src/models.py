"""Shared Pydantic models for the ScikitDocs starter.

Mirrors `project/src/models.py` for the most part. The local-only
deviation (M20 exercise 4) is that `QueryResponse.citations` and
`QueryResponse.confidence` carry Pydantic `Field` constraints — the
structured-output contract the learner wires at the gateway boundary
in Skill Pair 9.
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
    # TODO(m20-exercise-4): add Pydantic Field constraints — citations min_length=1, confidence between 0.0 and 1.0
    citations: list[Source]
    confidence: float
    model: str
    tokens: TokenUsage
    cost_usd: float
    cached: bool = False
    trace_id: str | None = None
    blocked_by: str | None = None
