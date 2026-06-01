# Cross-module invariants — ScikitDocs starter

The authoritative source is [`src/constants.py`](src/constants.py). This
file is a narrative wrapper that documents **why** each value was
chosen, with line references back to the source.

Modules MUST import these values from `src.constants` rather than
hardcoding the literals. The `make consistency-check` target (wired up) greps `content/implementation/` for forbidden bare
literals (`:8000`, `gpt-5.`, `ada-002`, `gpt-3.5`, etc.) — using the
constant guarantees a consistency-check pass.

## Twenty-one invariants

| Constant | Value | `constants.py` line | Why this value |
|---|---|---:|---|
| `SERVICE_PORT` | `8080` | 23 | Matches capstone `project/Makefile:35`. 8000 is reserved for Phoenix UI mirrors and learner-local apps. |
| `QUERY_ROUTE` | `/query` | 24 | Matches capstone `project/src/gateway/routes.py:61`. Cross-starter route parity. |
| `HEALTH_ROUTE` | `/health` | 25 | Matches capstone `project/src/gateway/routes.py:106` (NOT `/healthz`). Kubernetes-style convention deferred to match course precedent. |
| `MODEL_COMPLEX` | `gpt-4o` | 28 | Matches capstone `project/src/pricing.py:20`. Locked into `src/pricing.py` cost-table at scaffolding. |
| `MODEL_SIMPLE` | `gpt-4o-mini` | 29 | Matches capstone `project/src/pricing.py:21`. Used by the simple-vs-complex router (Module 18). |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | 30 | Matches capstone. 1536-dim, cheap, sufficient quality for documentation retrieval. |
| `EMBEDDING_DIM` | `1536` | 31 | OpenAI default for `text-embedding-3-small`. Asserted at Chroma collection create time (Module 05). |
| `DEFAULT_TOP_K` | `5` | 34 | Matches capstone `project/src/rag/pipeline.py:9`. Module 11 sweeps k∈{3,5,10} as a calibration exercise. |
| `CHUNK_TARGET_TOKENS` | `500` | 35 | RAG-over-docs research recommendation; balances precision (small) vs. context-completeness (large). |
| `CHUNK_OVERLAP_TOKENS` | `75` | 36 | 15% of target → preserves section continuity without much duplication. |
| `CONFIDENCE_THRESHOLD` | `0.7` | 37 | Matches capstone `project/src/config.py:54`. Below this the API surfaces "I'm not sure" framing. |
| `GENERATION_TEMPERATURE` | `0.2` | 38 | Docs assistant: low temperature for factual recall. Not zero so the generator can paraphrase. |
| `GOLDEN_SET_SIZE` | `30` | 41 | Matches capstone. Large enough for variance, small enough that judge-API cost stays under $1/sweep. |
| `JUDGE_TEMPERATURE` | `0.0` | 42 | Deterministic grading. The judge is a comparator, not a generator. |
| `CACHE_SIMILARITY_THRESHOLD` | `0.85` | 45 | Matches capstone semantic cache. Empirically tuned — see capstone history. |
| `COST_LOG_PATH` | `data/cost_log.jsonl` | 48 | Matches capstone. JSONL is append-only and grep-friendly. |
| `PHOENIX_PORT` | `6006` | 51 | Matches capstone. Phoenix's documented default; conflicts only with TensorBoard, which Course 2 doesn't use. |
| `PHOENIX_PROJECT_NAME` | `scikitdocs` | 52 | **Intentionally distinct** from capstone's `llm-ops-capstone` so learners running both side-by-side see separate trace streams. |
| `CLIENT_ID_HEADER` | `X-Client-Id` | 55 | Module 22 sticky-by-user A/B bucketing key. Cross-module contract — verified by `tests/test_smoke.py` post-scaffolding. |
| `OPENAI_BASE_URL_ENV` | `OPENAI_BASE_URL` | 58 | Matches memory `project_vocareum_deployment.md`. The openai SDK reads this directly via env. |
| `VOCAREUM_BASE_URL` | `https://openai.vocareum.com/v1` | 59 | Vocareum proxy URL — used when learners have a `voc-` API key. |

## Why these particular 21

These were the "hot patterns" identified in the deepen-plan review at
`docs/plans/2026-05-17-feat-rewrite-impl-modules-scikitdocs-altworkload-plan.md`
§"Cross-module consistency mechanism". Each one had a documented
historical drift across Wave 1-3 modules (or its capstone equivalent)
that the constants table prevents from recurring.

If you find a 22nd cross-module value drifting (e.g., two modules quote
different chunking overlaps in their concept walks), add it here as a
constants amendment via the **infra-amendment protocol**.
