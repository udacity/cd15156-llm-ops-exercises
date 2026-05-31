"""Exercise 1 — premium-tier conditional in the docbot system prompt."""

from src.generator import render_system_prompt
from src.models import Source


def _src(text: str) -> Source:
    return Source(doc_id="d", chunk_text=text, similarity_score=0.9)


def test_premium_tier_includes_mailing_list():
    out = render_system_prompt(
        [_src("LogisticRegression default penalty is l2.")],
        user_tier="premium",
    )
    assert "scikit-learn-help" in out or "mailman" in out


def test_standard_tier_omits_mailing_list():
    out = render_system_prompt(
        [_src("LogisticRegression default penalty is l2.")],
        user_tier="standard",
    )
    assert "scikit-learn-help" not in out
    assert "mailman" not in out
