# Constants reference — ScikitDocs starter

The authoritative source is [`src/constants.py`](src/constants.py). Import
these values from `src.constants` rather than hardcoding the literals — the
modules share them, so using the constant keeps everything consistent.

## Values

| Constant | Value | `constants.py` line | Why this value |
|---|---|---:|---|
| `SERVICE_PORT` | `8080` | 23 | FastAPI gateway port. 8000 is left free for the Phoenix UI and learner-local apps. |
| `QUERY_ROUTE` | `/query` | 24 | Primary query endpoint. |
| `HEALTH_ROUTE` | `/health` | 25 | Health-check endpoint (not `/healthz`). |
| `MODEL_COMPLEX` | `gpt-4o` | 28 | Default model for complex queries. |
| `MODEL_SIMPLE` | `gpt-4o-mini` | 29 | Cheaper model for the simple-vs-complex router. |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | 30 | 1536-dim, cheap, sufficient quality for documentation retrieval. |
| `EMBEDDING_DIM` | `1536` | 31 | OpenAI default for `text-embedding-3-small`. Asserted at Chroma collection-create time. |
| `DEFAULT_TOP_K` | `5` | 34 | Default retrieval depth. |
| `CHUNK_TARGET_TOKENS` | `500` | 35 | RAG-over-docs research recommendation; balances precision (small) vs. context-completeness (large). |
| `CHUNK_OVERLAP_TOKENS` | `75` | 36 | 15% of target → preserves section continuity without much duplication. |
| `CONFIDENCE_THRESHOLD` | `0.7` | 37 | Below this the API surfaces "I'm not sure" framing. |
| `GENERATION_TEMPERATURE` | `0.2` | 38 | Docs assistant: low temperature for factual recall. Not zero so the generator can paraphrase. |
| `GOLDEN_SET_SIZE` | `30` | 41 | Large enough for variance, small enough that judge-API cost stays under $1/sweep. |
| `JUDGE_TEMPERATURE` | `0.0` | 42 | Deterministic grading. The judge is a comparator, not a generator. |
| `CACHE_SIMILARITY_THRESHOLD` | `0.85` | 45 | Semantic cache hit threshold — empirically tuned. |
| `COST_LOG_PATH` | `data/cost_log.jsonl` | 48 | JSONL is append-only and grep-friendly. |
| `PHOENIX_PORT` | `6006` | 51 | Phoenix's documented default; conflicts only with TensorBoard, which this course doesn't use. |
| `PHOENIX_PROJECT_NAME` | `scikitdocs` | 52 | Trace-stream name for this workload. |
| `CLIENT_ID_HEADER` | `X-Client-Id` | 55 | Optional request header used as the A/B bucketing key. |
| `OPENAI_BASE_URL_ENV` | `OPENAI_BASE_URL` | 58 | The openai SDK reads this directly from the environment. |
| `VOCAREUM_BASE_URL` | `https://openai.vocareum.com/v1` | 59 | Vocareum proxy URL — used when you have a `voc-` API key. |
