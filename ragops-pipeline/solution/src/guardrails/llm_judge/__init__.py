"""LLM-as-judge output guard — hallucination scanning (REQ-072, M20).

The judge is a gpt-4o-mini call constructed at module load time with
the Vocareum-aware ``base_url`` discipline Module 18 covers (the
``settings.openai_base_url`` value comes from ``.env`` — empty string
when running against direct OpenAI, ``https://openai.vocareum.com/v1``
when running on a Vocareum ``voc-`` key).

The judge rubric (``prompts/judge.j2``) is scikit-learn-specific: an
answer is SUPPORTED if every cited API symbol, signature, parameter
name, or default value appears in the retrieved source chunks.
NOT_SUPPORTED if the answer cites a function/class/parameter the
sources do not mention. Replaces the medical-claim style rubric from
the pickleball capstone with one that maps to the documentation Q&A
workload.
"""

from src.guardrails.llm_judge.output_guards import check_hallucination

__all__ = ["check_hallucination"]
