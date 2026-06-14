# Side-by-side MiniLM wrapper that leaves the OpenAI embedder untouched
"""Local sentence-transformers embedder for the Module 05 swap exercise."""
from sentence_transformers import SentenceTransformer

# Loads all-MiniLM-L6-v2 once at import — first call warms the HF cache
_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Encodes the text with normalized embeddings and returns a plain Python list
def embed_query_st(text: str) -> list[float]:
    vec = _model.encode(text, normalize_embeddings=True)
    return vec.tolist()
