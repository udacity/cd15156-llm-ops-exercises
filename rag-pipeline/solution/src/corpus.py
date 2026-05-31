"""scikit-learn docs corpus loader (REQ-062).

Walks a pinned scikit-learn checkout, parses ``doc/**/*.rst`` with docutils,
and yields one dict per top-level section. Sphinx-specific roles
(``:func:``, ``:class:``, ``:ref:``, ``:doc:`` ...) are captured as
``{type, target}`` metadata via the stub-role pattern rather than triggering
an unknown-role error, so we never need a full Sphinx build environment.

The stub-role pattern is documented at
https://docutils.sourceforge.io/docs/howto/rst-roles.html and
https://docutils.sourceforge.io/docs/howto/rst-directives.html — we register
no-op handlers for the Sphinx-specific names docutils doesn't know about.

Known limitation: cross-references (``:ref:``, ``:doc:``) are captured but
not resolved — resolution would require a full Sphinx build. M07 (REQ-066)
discusses this trade-off when teaching RAG retrieval quality.
"""

import re
from collections.abc import Iterator
from pathlib import Path

import docutils.nodes
import tiktoken
from docutils import core, nodes
from docutils.parsers.rst import Directive, directives, roles

# Pinned scikit-learn source. Override the SHA when bumping the tag —
# data/CORPUS_VERSION records the resolved SHA per build.
SCIKIT_LEARN_TAG: str = "1.5.2"
SCIKIT_LEARN_REPO_URL: str = "https://github.com/scikit-learn/scikit-learn.git"
SCIKIT_LEARN_DOC_BASE_URL: str = "https://scikit-learn.org/stable/"

# RST subdirectories under ``doc/`` to ingest. ``auto_examples`` is
# sphinx-gallery-generated and only exists after a full Sphinx build —
# the loader skips it gracefully when absent.
DOC_SUBDIRS: tuple[str, ...] = ("modules", "tutorial", "auto_examples")

# Sphinx-specific text roles that have no docutils default. The stub
# role captures the target as metadata rather than raising
# "Unknown interpreted text role".
SPHINX_ROLES: tuple[str, ...] = (
    "func", "class", "meth", "attr", "mod", "ref", "doc", "obj",
    "data", "exc", "term", "rfc", "pep", "envvar", "command",
)

# Sphinx-specific directives. The stub directive accepts arbitrary
# arguments / content and emits an empty container so the surrounding
# RST parses cleanly.
SPHINX_DIRECTIVES: tuple[str, ...] = (
    "currentmodule", "module", "autosummary", "automodule", "autoclass",
    "autofunction", "automethod", "autoattribute", "autodata",
    "autoexception", "tabularcolumns", "topic", "rubric", "deprecated",
    "versionadded", "versionchanged", "seealso", "centered", "hlist",
    "plot", "include", "highlight", "literalinclude", "glossary",
    "index", "only", "toctree",
)

# Encoder for chunk size accounting. ``cl100k_base`` is the encoding for
# ``text-embedding-3-small`` / ``gpt-4o`` so token counts here match
# what OpenAI bills downstream.
_ENCODER = tiktoken.get_encoding("cl100k_base")


def _xref_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    """Stub Sphinx text-role: capture ``:role:`label <target>``` as metadata.

    Returns a ``literal`` node with ``xref_type`` and ``xref_target``
    attributes set, so :func:`_extract_xrefs` can collect them later.
    Sphinx supports ``label <target>`` syntax to display ``label`` while
    linking to ``target`` — we honour both shapes.
    """
    del rawtext, lineno, inliner, options, content
    target = text
    label = text
    match = re.match(r"^(.*?)\s*<(.+)>\s*$", text)
    if match:
        label = match.group(1).strip()
        target = match.group(2).strip()
    if not label:
        label = target
    node = nodes.literal(label, label, classes=[f"xref-{name}"])
    node["xref_type"] = name
    node["xref_target"] = target
    return [node], []


class _NoopDirective(Directive):
    """Stub Sphinx directive: parse content into a container, discard options."""

    has_content = True
    required_arguments = 0
    optional_arguments = 100
    final_argument_whitespace = True
    option_spec: dict = {}

    def run(self) -> list:
        if not self.content:
            return []
        container = nodes.container()
        self.state.nested_parse(self.content, self.content_offset, container)
        return [container]


_stubs_registered: bool = False


def _register_stubs() -> None:
    """Idempotently register no-op handlers for Sphinx roles + directives."""
    global _stubs_registered
    if _stubs_registered:
        return
    for role in SPHINX_ROLES:
        roles.register_local_role(role, _xref_role)
    for directive in SPHINX_DIRECTIVES:
        directives.register_directive(directive, _NoopDirective)
    _stubs_registered = True


def _parse_rst(path: Path) -> docutils.nodes.document:
    """Parse one RST file. ``report_level=4`` silences Sphinx-role warnings.

    ``doctitle_xform=False`` keeps a single top-level section as a real
    ``nodes.section`` instead of promoting its title to the doctree
    title and discarding the wrapper — which would lose the section
    from our walk.
    """
    _register_stubs()
    text = path.read_text(encoding="utf-8")
    return core.publish_doctree(
        text,
        settings_overrides={
            "report_level": 4,
            "halt_level": 5,
            "warning_stream": None,
            "embed_stylesheet": False,
            "input_encoding": "utf-8",
            "doctitle_xform": False,
            "sectsubtitle_xform": False,
        },
    )


def _section_title(section: nodes.section) -> str:
    for child in section.children:
        if isinstance(child, nodes.title):
            return child.astext()
    return ""


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug


def _detect_code(section: nodes.section) -> tuple[bool, list[str]]:
    """Return (has_code, sorted unique languages) for direct-child code blocks."""
    has_code = False
    languages: set[str] = set()
    for block in section.findall(nodes.literal_block):
        # Skip code blocks owned by a nested sub-section — those belong
        # to that child's chunk, not this one.
        if _belongs_to_descendant_section(block, section):
            continue
        has_code = True
        languages.add(block.get("language") or "python")
    return has_code, sorted(languages)


def _belongs_to_descendant_section(node: nodes.Node, ancestor: nodes.section) -> bool:
    """True if ``node`` lives inside a sub-section of ``ancestor``."""
    parent = node.parent
    while parent is not None and parent is not ancestor:
        if isinstance(parent, nodes.section):
            return True
        parent = parent.parent
    return False


def _extract_xrefs(section: nodes.section) -> list[dict]:
    """Collect ``{type, target}`` dicts from stub-role literal nodes."""
    xrefs: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for node in section.findall(nodes.literal):
        if "xref_type" not in node.attributes:
            continue
        if _belongs_to_descendant_section(node, section):
            continue
        key = (node["xref_type"], node["xref_target"])
        if key in seen:
            continue
        seen.add(key)
        xrefs.append({"type": node["xref_type"], "target": node["xref_target"]})
    return xrefs


def _section_body_text(section: nodes.section) -> str:
    """Concatenated text of this section's direct children, skipping nested sections."""
    parts: list[str] = []
    for child in section.children:
        if isinstance(child, (nodes.title, nodes.section)):
            continue
        text = child.astext()
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts).strip()


def _rel_path_for_url(rel_path: Path) -> str:
    parts = list(rel_path.with_suffix(".html").parts)
    if parts and parts[0] == "doc":
        parts = parts[1:]
    return "/".join(parts)


def _rel_path_for_doc_id(rel_path: Path) -> str:
    parts = list(rel_path.with_suffix("").parts)
    if parts and parts[0] == "doc":
        parts = parts[1:]
    return ".".join(parts)


def _section_url(rel_path: Path, anchor: str) -> str:
    base = SCIKIT_LEARN_DOC_BASE_URL + _rel_path_for_url(rel_path)
    return f"{base}#{anchor}" if anchor else base


def _section_doc_id(rel_path: Path, anchor: str) -> str:
    base = _rel_path_for_doc_id(rel_path)
    return f"{base}.{anchor}" if anchor else base


def token_count(text: str) -> int:
    """Tokens under the ``text-embedding-3-small`` / ``gpt-4o`` encoding."""
    return len(_ENCODER.encode(text, disallowed_special=()))


def iter_sections_in_doctree(
    doctree: docutils.nodes.document, rel_path: Path
) -> Iterator[dict]:
    """Yield one dict per section in ``doctree`` (recursive, depth-first)."""

    def walk(parent: nodes.Node, ancestors: list[str]) -> Iterator[dict]:
        for child in parent.children:
            if not isinstance(child, nodes.section):
                continue
            title = _section_title(child)
            anchor = _slugify(title)
            body = _section_body_text(child)
            if body:
                has_code, languages = _detect_code(child)
                yield {
                    "text": body,
                    "doc_id": _section_doc_id(rel_path, anchor),
                    "metadata": {
                        "source_path": str(rel_path),
                        "section_title": title,
                        "section_path": " > ".join(ancestors + [title]),
                        "url": _section_url(rel_path, anchor),
                        "has_code": has_code,
                        "code_languages": languages,
                        "xrefs": _extract_xrefs(child),
                    },
                }
            yield from walk(child, ancestors + [title])

    yield from walk(doctree, [])


def load_corpus(repo_path: Path, version_sha: str) -> Iterator[dict]:
    """Walk a pinned scikit-learn repo's ``doc/`` tree, yield one dict per RST section.

    Each yielded dict has:
        text       — section body (RST stripped to plain text)
        metadata   — {source_path, section_title, section_path, url,
                      has_code, code_languages, xrefs, scikit_learn_sha}
        doc_id     — stable id derived from path + slugified section title

    Args:
        repo_path:    Path to the cloned scikit-learn repo root.
        version_sha:  Git SHA pin (recorded in data/CORPUS_VERSION).

    Yields:
        dict — one per RST section across ``doc/modules/``,
        ``doc/tutorial/``, and (if present) ``doc/auto_examples/``.

    Raises:
        FileNotFoundError: if ``repo_path/doc/`` does not exist.
    """
    doc_root = repo_path / "doc"
    if not doc_root.exists():
        raise FileNotFoundError(f"No doc/ tree under {repo_path}")

    rst_files: list[Path] = []
    for sub in DOC_SUBDIRS:
        subdir = doc_root / sub
        if subdir.exists():
            rst_files.extend(sorted(subdir.rglob("*.rst")))

    for rst_path in rst_files:
        rel_path = rst_path.relative_to(repo_path)
        try:
            doctree = _parse_rst(rst_path)
        except Exception:
            # An individual broken file shouldn't tank the whole ingest.
            # Real corruption shows up as 0 chunks for that file.
            continue
        for section in iter_sections_in_doctree(doctree, rel_path):
            section["metadata"]["scikit_learn_sha"] = version_sha
            yield section
