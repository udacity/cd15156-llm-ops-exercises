"""Section-header chunker for scikit-learn doc sections (Module 05).

Takes one section dict from :func:`src.corpus.load_corpus` and returns a
list of chunk dicts ready for embedding. Strategy:

- Respect section-header boundaries (corpus.py already yields one dict
  per section, so the section boundary is the input boundary).
- Sections at-or-under ``constants.CHUNK_TARGET_TOKENS`` pass through
  whole. Longer sections split at paragraph boundaries with
  ``constants.CHUNK_OVERLAP_TOKENS`` overlap.
- Chunks shorter than ``_CHUNK_MIN_TOKENS`` are dropped — they're noise
  for retrieval (one-line residual headings, "See also" stubs).
- ``has_code``, ``code_languages``, ``xrefs`` propagate from the source
  doc's metadata unchanged: a chunk inherits the section's code/xref
  flags even if a particular split piece doesn't contain the code block
  itself, because retrieval-by-metadata downstream (Module 11) filters at
  section granularity.

The chunk-size knobs match :mod:`scripts.load_data` so a chunk built
here is byte-identical to one built by the load-time path.
"""

from src import constants, corpus

# Sections shorter than this are skipped as retrieval noise. Matches
# load_data.py's CHUNK_MIN_TOKENS.
_CHUNK_MIN_TOKENS: int = 50

# Hard ceiling above which we split. Slightly above CHUNK_TARGET_TOKENS
# so we don't aggressively split sections that are just a hair over.
_CHUNK_MAX_TOKENS: int = 512


def _split_long_text(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Split ``text`` at paragraph boundaries, respecting ``max_tokens``."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for paragraph in paragraphs:
        ptokens = corpus.token_count(paragraph)
        if ptokens > max_tokens:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_tokens = 0
            chunks.append(paragraph)
            continue
        if current_tokens + ptokens > max_tokens:
            chunks.append("\n\n".join(current))
            tail = current[-1] if current else ""
            tail_tokens = corpus.token_count(tail)
            if tail_tokens <= overlap:
                current = [tail, paragraph]
                current_tokens = tail_tokens + ptokens
            else:
                current = [paragraph]
                current_tokens = ptokens
        else:
            current.append(paragraph)
            current_tokens += ptokens
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_doc(doc: dict) -> list[dict]:
    """Split a doc-section dict into one or more chunk dicts.

    Args:
        doc: A dict from :func:`src.corpus.load_corpus` — has ``text``,
            ``doc_id``, and ``metadata`` keys.

    Returns:
        list[dict] — each with ``text``, ``metadata``, ``chunk_id``.
        Empty list if the section is below ``_CHUNK_MIN_TOKENS``.
    """
    text = doc["text"]
    base_id = doc["doc_id"]
    base_metadata = dict(doc["metadata"])
    tokens = corpus.token_count(text)
    if tokens < _CHUNK_MIN_TOKENS:
        return []
    if tokens <= _CHUNK_MAX_TOKENS:
        return [{
            "text": text,
            "metadata": base_metadata,
            "chunk_id": base_id,
        }]
    pieces = _split_long_text(text, _CHUNK_MAX_TOKENS, constants.CHUNK_OVERLAP_TOKENS)
    chunks: list[dict] = []
    for idx, piece in enumerate(pieces):
        if corpus.token_count(piece) < _CHUNK_MIN_TOKENS:
            continue
        chunks.append({
            "text": piece,
            "metadata": dict(base_metadata),
            "chunk_id": f"{base_id}.p{idx}",
        })
    return chunks
