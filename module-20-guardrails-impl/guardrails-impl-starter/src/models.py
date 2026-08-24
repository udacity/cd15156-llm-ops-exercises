"""Shared Pydantic models for the ScikitDocs starter.

These are the shared type shapes every layer imports. The local-only
deviation (exercise 4) is the `QueryResponseValidator` companion model
the learner builds at the bottom of this file — the structured-output
guard wired at the gateway boundary in the output-validator exercise.
`QueryResponse` itself stays unconstrained.
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


# TODO(m20-exercise-4): build the QueryResponseValidator companion model — sources min_length=1, confidence between 0.0 and 1.0
