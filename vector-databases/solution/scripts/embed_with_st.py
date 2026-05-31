"""Local sentence-transformers embedder for the M05 swap exercise."""
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_query_st(text: str) -> list[float]:
    vec = _model.encode(text, normalize_embeddings=True)
    return vec.tolist()
