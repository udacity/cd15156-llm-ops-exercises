# TODO(m05-ex3): write scripts/embed_with_st.py — side-by-side MiniLM wrapper so
# the OpenAI embedder stays untouched for the swap comparison.
"""Local sentence-transformers embedder for the M05 swap exercise."""
from sentence_transformers import SentenceTransformer

# TODO(m05-ex3): load all-MiniLM-L6-v2 once at import; first import warms the HF cache.
_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# TODO(m05-ex3): mirror embedder.embed_query — encode with normalize_embeddings=True
# (the cosine-pinned store needs unit vectors) and return a plain Python list.
def embed_query_st(text: str) -> list[float]:
    vec = _model.encode(text, normalize_embeddings=True)
    return vec.tolist()
