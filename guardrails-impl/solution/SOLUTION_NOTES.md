# Module 20 Solution Notes

This `solution/` is the ScikitDocs starter with **Exercise 4 (Pydantic `QueryResponse` validator at the gateway boundary)** already applied. The other three exercises produce per-learner artifacts (added input guard, calibration markdown report, cost-amplification table + burst-test script output) rather than code that ships in the repo; reference outputs are described below.

## Exercise 4 — Pydantic output validator (code-in-repo)

The change is bracketed by `# TODO(m20-exercise-4)-start` / `-end` markers so a learner can `grep` the seam.

- `src/models.py` — `QueryResponse` itself carries the constraints: `citations: list[Source] = Field(..., min_length=1)` and `confidence: float = Field(..., ge=0.0, le=1.0)`. `Field` is imported from `pydantic` alongside `BaseModel`. The field rename (`sources` → `citations`) is local to this m20 exercise; other modules in this course keep `sources`.
- `src/gateway/routes.py` — `try/except ValidationError` block at the boundary, after the hallucination check, just before the final `return response`. On `ValidationError`, returns `JSONResponse(status_code=502, content={"detail": "output_validation_failed", "field": str(exc.errors()[0]['loc'][0])})`. Imports widened to bring in `JSONResponse` and `ValidationError`. Return type widened to `QueryResponse | JSONResponse`.
- `tests/test_gateway_output_validator.py` — the two tests the exercise asks the learner to author: (1) well-formed response → 200, (2) `citations=[]` → 502 with `field=="citations"`. Both mock `route_query` and `check_hallucination` so the test exercises only the validator seam, and `reset_rate_limit_state()` runs at the top of each test so the LLM10 bucket doesn't leak between runs. The citation-stripped fixture uses `QueryResponse.model_construct(...)` to bypass the constructor's own validation; the test is about the boundary re-validation, not the constructor.

Verify the wire-up:

```bash
grep -n "TODO(m20-exercise-4)" src/models.py src/gateway/routes.py
uv run pytest tests/test_gateway_output_validator.py -v
```

## Exercise 1 — New input guard (per-learner code)

The starter's `src/guardrails/input_guards.py` is unchanged here; the exercise asks the learner to pick **one** of three options (invisible-Unicode, custom-domain PII pattern `SDP-\d{6}`, new system-prompt-leak phrasing) and add it. Reference shapes:

- **Option A (invisible-Unicode):** new `detect_invisible_unicode(text: str) -> str | None` in `src/guardrails/input_guards.py` that flags zero-width spaces (U+200B), zero-width joiners (U+200D), bidi override marks (U+202E), and runs of soft hyphens (U+00AD). Wire into `src/gateway/routes.py` between the rate-limit check and the prompt-injection check (see INSTRUCTIONS.md → Exercise 1 for the integration point).
- **Option B (SDP-\d{6}):** new key in `PII_PATTERNS` + `PII_REDACTIONS`. The change is implicit — `detect_pii` iterates the dict, so no route edit is needed.
- **Option C (new leak phrasing):** new key in `SYSTEM_PROMPT_LEAK_PATTERNS`. Same implicit integration — `detect_system_prompt_leak` iterates the dict.

Acceptance is a parametrised pytest case in `tests/test_guardrails.py` (one trigger, one non-trigger, one edge case) plus a live `curl` against `/query` on `:8080` showing the expected `blocked_by` reason (Options A/C) or the redacted-question pass-through (Option B).

## Exercise 2 — LLM-judge calibration (per-learner markdown)

Expected deliverable: `exercises/M20/judge-calibration-sweep.md` containing two confusion matrices (one per configuration the learner sweeps), a one-paragraph interpretation, and a recommendation with rationale. The sweep cohort is ten queries: five grounded scikit-learn API questions (true-negatives) + five subtly-wrong-API questions (true-positives). Reference table shape:

```
| config | TP | FN | TN | FP | FP rate | FN rate |
|---|---|---|---|---|---|---|
| baseline rubric, gpt-4o-mini, T=0.0 | 4 | 1 | 4 | 1 | 0.20 | 0.20 |
| stricter rubric, gpt-4o-mini, T=0.0 | 5 | 0 | 3 | 2 | 0.40 | 0.00 |
```

Numbers will drift across runs (small sample, judge stochasticity even at T=0). The recommendation should pick a point on the FP/FN curve and name the cost asymmetry for documentation Q&A: a wrong-API answer that escapes the judge is a learner copying a deprecated symbol into production code (high cost); a grounded answer that the judge flags is a learner who gets `SAFE_FILTERED_MESSAGE` and re-asks (lower cost). The honest hedge: ten queries is underpowered — the procedure is the lesson, the threshold is the artifact.

Bonus: rerun the cohort through LLM Guard's `FactualConsistency` NLI scanner (importable from `llm_guard.output_scanners`) and add a third matrix to the report.

## Exercise 3 — LLM10 cost-amplification + burst block (per-learner table + script + patched generator)

Three deliverables:

1. **Cost-amplification table.** Run two ten-request cohorts through `/query` (`X-Client-Id: cohort-baseline` for baseline, `cohort-long` for long-generation prompt like "Write a thorough tutorial covering every parameter of `RandomForestClassifier`..."). Read `data/cost_log.jsonl`, compute median and p95 `cost_usd` per cohort, divide long-p95 by baseline-p95 → amplification factor. Typical result on `gpt-4o-mini`: 3×–8×. CVE-2025-53773 reported up to 20× on GitHub Copilot; ScikitDocs is a simpler workload, so the number is smaller but the shape is the same.

2. **`max_tokens` wired into `src/generator.py`.** Add `max_tokens: int | None = None` kwarg to `generate(...)`, pass through to `openai.chat.completions.create(max_tokens=max_tokens)`, and forward `MAX_OUTPUT_TOKENS` from `src.guardrails.rate_limit` in `src.pipeline.run_pipeline` and `src.gateway.router.route_query`. After the patch, re-run the long-generation cohort and confirm the p95 cost drops to the cap-implied ceiling.

3. **Burst-test script + output.** Shell or Python script that fires 25 sequential requests against `/query` with `X-Client-Id: burst-test`. Requests 1–20 return normal `QueryResponse` bodies; requests 21–25 return HTTP 200 with `answer == SAFE_BLOCKED_MESSAGE` and `blocked_by == "unbounded_consumption: rate limit exceeded (20 requests per 60s)"`. After 60 seconds of idle, one more request should pass — the bucket has rolled forward.

Bonus: tighten `RATE_LIMIT_REQUESTS = 5` and `RATE_LIMIT_WINDOW_SECONDS = 30` in `src/guardrails/rate_limit.py`, rerun the burst test, and report the new threshold. The point is that the defaults are starting values keyed off a cost budget, not constants of nature.

## KNOWN-LIMITATIONs

- **Exercise 1/2/3 code not authored in `solution/`.** Exercises 1 (input guard), 2 (calibration report), and 3 (cost-amplification + `max_tokens` patch + burst script) produce per-learner deliverables — the exercise menu offers three input-guard options, the calibration knob is a learner choice, and the burst-script implementation can be bash or Python. SOLUTION_NOTES.md above describes the expected shape and acceptance criteria; the solution repo only carries the Exercise 4 code (which has one canonical answer).
- **Live-route tests need OpenAI keys + heavy NLI models.** `tests/test_guardrails.py` is skipped in the central verification pass because the DeBERTa prompt-injection model (~250 MB), the `dslim/bert-base-NER` Presidio backend, and `en_core_web_sm` all need to be downloadable on the verification box. The Exercise 4 tests in `tests/test_gateway_output_validator.py` mock `route_query` and `check_hallucination` so they run in well under a second without any model load.
- **`max_tokens` plumbing for Exercise 3 not pre-applied in `solution/`.** Exercise 3 step 2 explicitly asks the learner to patch `src/generator.py`. Pre-applying it would obviate the exercise. The `MAX_OUTPUT_TOKENS = 1024` constant exists in `src/guardrails/rate_limit.py`; the learner's job is to thread it through.
