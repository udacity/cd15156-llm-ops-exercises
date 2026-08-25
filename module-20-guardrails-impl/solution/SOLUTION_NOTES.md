# Module 20 Solution Notes

These notes accompany the ScikitDocs solution, which ships a reference solution for **all four exercises** under `TODO(m20-...)` markers, with matching stubs in the starter. A stuck learner greps the marker in `starter/` and finds the finished version in `solution/`. You still write your own; the reference is there when you need it. What ships per exercise is described below.

## Exercise 4 — Pydantic output validator (code-in-repo)

Each authored block sits under a single-line `# TODO(m20-exercise-4)` marker so a learner can `grep` the seam.

- `src/models.py` — `QueryResponse` stays unconstrained; the new `QueryResponseValidator` companion model at the bottom of the file carries the contract: `answer: str`, `sources: list[Source] = Field(..., min_length=1)`, `confidence: float = Field(..., ge=0.0, le=1.0)`. `Field` is imported from `pydantic` alongside `BaseModel`.
- `src/gateway/routes.py` — `try/except ValidationError` block at the boundary, after the hallucination check, just before the final `return response`, validating `QueryResponseValidator.model_validate(response.model_dump())`. On `ValidationError`, returns `JSONResponse(status_code=502, content={"detail": "output_validation_failed", "field": str(exc.errors()[0]['loc'][0])})` — 502 because a contract violation here is a bug on our side, not a client error. Imports widened to bring in `JSONResponse`, `ValidationError`, and `QueryResponseValidator`. Return type widened to `QueryResponse | JSONResponse`.
- `tests/test_gateway_output_validator.py` — the two tests the exercise asks the learner to author: (1) well-formed response → 200, (2) `sources=[]` → 502 with `field=="sources"`. Both mock `route_query` and `check_hallucination` so the test exercises only the validator seam, and `reset_rate_limit_state()` runs at the top of each test so the LLM10 bucket doesn't leak between runs. The source-stripped fixture uses the plain `QueryResponse(...)` constructor — the base model accepts `sources=[]`; only the boundary validator rejects it.

Verify the wire-up:

```bash
grep -n "TODO(m20-exercise-4)" src/models.py src/gateway/routes.py
uv run pytest tests/test_gateway_output_validator.py -v
```

## Exercise 1 — New input guard (canonical Option A shipped)

This solution ships the canonical **Option A** implementation under `TODO(m20-exercise-1)` markers (guard function in `src/guardrails/input_guards.py`, wiring in `src/gateway/routes.py`, parametrised test in `tests/test_guardrails.py`); the starter carries the same markers with the code stripped, per the course-wide TODO convention. The exercise asks the learner to pick **one** of three options (invisible-Unicode, custom-domain PII pattern `SDP-\d{6}`, new system-prompt-leak phrasing). Reference shapes:

- **Option A (invisible-Unicode):** new `detect_invisible_unicode(text: str) -> str | None` in `src/guardrails/input_guards.py` that flags zero-width spaces (U+200B), zero-width joiners (U+200D), bidi override marks (U+202E), and runs of soft hyphens (U+00AD). Wire into `src/gateway/routes.py` between the rate-limit check and the prompt-injection check. **This is the implementation shipped here.**
- **Option B (SDP-\d{6}):** new key in `PII_PATTERNS` + `PII_REDACTIONS`. The change is implicit — `detect_pii` iterates the dict, so no route edit is needed.
- **Option C (new leak phrasing):** new key in `SYSTEM_PROMPT_LEAK_PATTERNS`. Same implicit integration — `detect_system_prompt_leak` iterates the dict.

Acceptance is a parametrised pytest case in `tests/test_guardrails.py` (one trigger, one non-trigger, one edge case) plus a live `curl` against `/query` on `:8080` showing the expected `blocked_by` reason (Options A/C) or the redacted-question pass-through (Option B).

## Exercise 2 — LLM-judge calibration (reference sweep + report shipped)

Shipped reference: `scripts/judge_calibration_sweep.py` runs the cohort and prints a confusion matrix per model, and `exercises/judge-calibration-sweep.md` is the written report containing two confusion matrices (one per configuration the learner sweeps), a one-paragraph interpretation, and a recommendation with rationale. The shipped sweep varies the **judge model** (`gpt-4o-mini` vs `gpt-4o`) on the same rubric. The sweep cohort is ten queries: five grounded scikit-learn API questions (true-negatives) + five subtly-wrong-API questions (true-positives). Reference table shape:

```
| config | TP | FN | TN | FP | FP rate | FN rate |
|---|---|---|---|---|---|---|
| gpt-4o-mini, baseline rubric, T=0.0 | 1 | 4 | 5 | 0 | 0.00 | 0.80 |
| gpt-4o, baseline rubric, T=0.0 | 0 | 5 | 4 | 1 | 0.20 | 1.00 |
```

Numbers will drift across runs (small sample, judge stochasticity even at T=0). The recommendation should pick a point on the FP/FN curve and name the cost asymmetry for documentation Q&A: a wrong-API answer that escapes the judge is a learner copying a deprecated symbol into production code (high cost); a grounded answer that the judge flags is a learner who gets `SAFE_FILTERED_MESSAGE` and re-asks (lower cost). The honest hedge: ten queries is underpowered — the procedure is the lesson, the threshold is the artifact.

Bonus: rerun the cohort through LLM Guard's `FactualConsistency` NLI scanner (importable from `llm_guard.output_scanners`) and add a third matrix to the report.

## Exercise 3 — LLM10 cost-amplification + burst block (reference table + script shipped; `max_tokens` plumbing shipped)

Three deliverables:

1. **Cost-amplification table.** Run two ten-request cohorts through `/query` (`X-Client-Id: cohort-baseline` for baseline, `cohort-long` for long-generation prompt like "Write a thorough tutorial covering every parameter of `RandomForestClassifier`..."). Read `data/cost_log.jsonl`, compute median and p95 `cost_usd` per cohort, divide long-p95 by baseline-p95 → amplification factor. Measured result on `gpt-4o-mini`: about 1.6×, small because the grounded generator rarely runs past ~700 tokens (far below CVE-2025-53773's up-to-20× on the GitHub Copilot code-generation workload). The reference measurement ships at `scripts/cost_amplification.py` (fires the cohorts, derives an output cap from the uncapped answer-length distribution at mean + 1 sd, and shows it clip the tail) and the captured result at `exercises/cost-amplification.md`.

2. **`max_tokens` wired into the generator — shipped here under `TODO(m20-exercise-3)` markers** at the four seams: `generate(...)` gains a `max_tokens: int | None = None` kwarg and forwards it through the `_call_chat_completions` retry wrapper to `openai.chat.completions.create`, `src.pipeline.run_pipeline` and `src.tracing.traced_pipeline` thread it through (the live gateway path runs through `traced_pipeline`, so skipping it would leave `/query` uncapped), and `src.gateway.router.route_query` forwards `MAX_OUTPUT_TOKENS` from `src.guardrails.rate_limit`. With the plumbing in, re-run the long-generation cohort and confirm the p95 cost drops to the cap-implied ceiling.

3. **Burst-test script + output (reference at `scripts/burst_test.py`).** Shell or Python script that fires 25 sequential requests against `/query` with `X-Client-Id: burst-test`. Requests 1–20 return normal `QueryResponse` bodies; requests 21–25 return HTTP 200 with `answer == SAFE_BLOCKED_MESSAGE` and `blocked_by == "unbounded_consumption: rate limit exceeded (20 requests per 60s)"`. After 60 seconds of idle, one more request should pass — the bucket has rolled forward.

Bonus: tighten `RATE_LIMIT_REQUESTS = 5` and `RATE_LIMIT_WINDOW_SECONDS = 30` in `src/guardrails/rate_limit.py`, rerun the burst test, and report the new threshold. The point is that the defaults are starting values keyed off a cost budget, not constants of nature.

## KNOWN-LIMITATIONs

- **All four exercises ship a reference solution** under `TODO(m20-...)` markers, matching the course-wide convention: the starter ships the markers (existing code stripped for Ex1/Ex3-plumbing/Ex4, or a scaffold for the from-scratch scripts `scripts/judge_calibration_sweep.py` and `scripts/burst_test.py`), and the solution ships the markers plus the implementation. You still author your own; the Option B/C guard variants are described above as reference shapes. The live-key outputs (the sweep's confusion matrices, the cost-amplification numbers) are representative captures, not fixed answers: rerun with your key and read your own.
- **Live-route tests need OpenAI keys + heavy NLI models.** `tests/test_guardrails.py` is skipped in the central verification pass because the DeBERTa prompt-injection model (~700 MB), the `dslim/bert-base-NER` Presidio backend, and `en_core_web_sm` all need to be downloadable on the verification box. The Exercise 4 tests in `tests/test_gateway_output_validator.py` mock `route_query` and `check_hallucination` so they run in well under a second without any model load.
