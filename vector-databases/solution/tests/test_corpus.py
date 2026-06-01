"""5-chunk hand-authored smoke test for src.corpus (the initial scaffolding acceptance).

A synthetic ``cross_validation.rst`` exercises the parser end-to-end:
section headers (level 1 + 2 + 3), :func:/:class:/:ref:/:doc: roles,
``code-block`` directives, and Sphinx-specific ``currentmodule`` +
``seealso`` + ``versionadded`` directives that must be silently stubbed.

The full ``make load-data`` happy path (clone + embed + upsert) needs an
OpenAI key and network, so it's exercised manually on the Workspace
rather than here. This file proves parser correctness in isolation.
"""

from pathlib import Path

import pytest

from src import corpus

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def sections() -> list[dict]:
    rel_path = Path("doc/modules/cross_validation.rst")
    doctree = corpus._parse_rst(FIXTURE_DIR / "sample_doctree.rst")
    return list(corpus.iter_sections_in_doctree(doctree, rel_path))


def test_yields_at_least_five_sections(sections: list[dict]) -> None:
    assert len(sections) >= 5, f"expected ≥5 sections, got {len(sections)}"


def test_section_shape(sections: list[dict]) -> None:
    keys = {"text", "doc_id", "metadata"}
    for section in sections:
        assert set(section.keys()) >= keys
        assert isinstance(section["text"], str) and section["text"].strip()
        assert isinstance(section["doc_id"], str)
        meta = section["metadata"]
        assert set(meta.keys()) >= {
            "source_path", "section_title", "section_path",
            "url", "has_code", "code_languages", "xrefs",
        }


def test_top_level_section_is_first(sections: list[dict]) -> None:
    top = sections[0]
    assert top["metadata"]["section_title"] == (
        "Cross-validation: evaluating estimator performance"
    )
    assert top["metadata"]["section_path"] == (
        "Cross-validation: evaluating estimator performance"
    )


def test_doc_id_uses_dotted_path(sections: list[dict]) -> None:
    top = sections[0]
    assert top["doc_id"].startswith("modules.cross_validation.")
    assert "/" not in top["doc_id"]


def test_url_points_to_scikit_learn_org(sections: list[dict]) -> None:
    top = sections[0]
    url = top["metadata"]["url"]
    assert url.startswith("https://scikit-learn.org/stable/")
    assert "modules/cross_validation.html" in url
    assert "#" in url  # anchor present


def test_code_block_detection(sections: list[dict]) -> None:
    by_title = {s["metadata"]["section_title"]: s for s in sections}
    assert "The basic K-Fold idiom" in by_title
    code_section = by_title["The basic K-Fold idiom"]
    assert code_section["metadata"]["has_code"] is True
    assert "python" in code_section["metadata"]["code_languages"]


def test_section_without_code_block_flagged_false(sections: list[dict]) -> None:
    by_title = {s["metadata"]["section_title"]: s for s in sections}
    pitfalls = by_title["Caveats and pitfalls"]
    assert pitfalls["metadata"]["has_code"] is False
    assert pitfalls["metadata"]["code_languages"] == []


def test_xrefs_captured(sections: list[dict]) -> None:
    by_title = {s["metadata"]["section_title"]: s for s in sections}
    top = by_title["Cross-validation: evaluating estimator performance"]
    xref_targets = {(x["type"], x["target"]) for x in top["metadata"]["xrefs"]}
    # Stub roles must capture the targets even though docutils doesn't
    # know what to do with Sphinx-specific role names.
    assert ("ref", "model-evaluation") in xref_targets
    assert ("class", "KFold") in xref_targets
    assert ("class", "StratifiedKFold") in xref_targets


def test_xrefs_label_target_syntax(sections: list[dict]) -> None:
    # Sphinx's ``label <target>`` syntax should resolve to the target.
    for section in sections:
        for xref in section["metadata"]["xrefs"]:
            assert "<" not in xref["target"]
            assert ">" not in xref["target"]


def test_subsection_breadcrumb(sections: list[dict]) -> None:
    by_title = {s["metadata"]["section_title"]: s for s in sections}
    repeated = by_title.get("Repeated cross-validation")
    assert repeated is not None, "expected nested section to be yielded"
    path = repeated["metadata"]["section_path"]
    assert path.startswith(
        "Cross-validation: evaluating estimator performance > "
        "Stratification considerations > "
    )


def test_sphinx_directives_do_not_crash(sections: list[dict]) -> None:
    # ``currentmodule``, ``seealso``, ``versionadded`` are Sphinx-only.
    # The stub directives must let the surrounding RST parse cleanly.
    titles = {s["metadata"]["section_title"] for s in sections}
    assert "Caveats and pitfalls" in titles  # parser made it to the end


def test_load_corpus_missing_repo_raises() -> None:
    with pytest.raises(FileNotFoundError):
        list(corpus.load_corpus(Path("/nonexistent-scikit-cache"), "deadbeef"))


def test_token_count_matches_tiktoken() -> None:
    # Sanity check the encoder is the one we expect.
    assert corpus.token_count("hello world") == 2
