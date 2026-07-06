# TODO(m05-ex3): write scripts/embed_with_st.py, side-by-side MiniLM wrapper so
# the OpenAI embedder stays untouched for the swap comparison.
"""Local sentence-transformers embedder for the Module 05 swap exercise."""
from sentence_transformers import SentenceTransformer

# Loads all-MiniLM-L6-v2 once at import — first call warms the HF cache
_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Encodes the text with normalized embeddings and returns a plain Python list
def embed_query_st(text: str) -> list[float]:
    vec = _model.encode(text, normalize_embeddings=True)
    return vec.tolist()
