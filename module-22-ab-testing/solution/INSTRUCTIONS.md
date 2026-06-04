# Module 22 — Conduct Prompt A/B Tests with Python Feature Flags

## Setup

This starter is the ScikitDocs RAG app — a Q&A assistant for the scikit-learn library — with the full instrumentation stack already wired: prompt loader and Jinja templates, Chroma vector store, RAG pipeline (`run_pipeline`), Phoenix tracing, RAGAS evaluation harness, cost monitoring (`src/pricing.py` + `src/cost/`), semantic answer cache, FastAPI gateway, guardrails, and a streaming endpoint. The A/B testing primitives are already in place at `src/optimization/ab.py` (the three functions `pick_variant`, `call_with_variant`, `log_assignment`), the two prompt variant files at `prompts/docbot_system_A.j2` and `_B.j2` are pre-shipped, and the analyzer at `scripts/ab_analyze.py` plus the `make ab-analyze` target are wired. In this module you will write one new script — `scripts/ab_simulate.py` — that fires 200 sticky-by-user calls through the two variants, then run `make ab-analyze` to read the chi-squared result and per-variant cost/latency aggregates honestly.

Bring up the corpus before you start:

```bash
uv sync
cp .env.example .env          # add your OPENAI_API_KEY (or Vocareum voc- key)
make load-data                # ~45–60s cold, ~5s warm; ~$0.10 in embeddings
```

Smoke-check the pipeline before any A/B work:

```bash
uv run python -c "from src.pipeline import run_pipeline; print(run_pipeline('What kernel does SVC use by default?').answer[:80])"
```

If that returns a grounded answer about `rbf`, you are ready. Follow the demo walkthrough first, then work through the three exercises in order.

---

> A walkthrough of this codebase is in DEMO.md.

# ... extract answer, usage, compute cost ...
return answer, usage, cost_usd, latency_ms
```

Two design choices worth naming. The `base_url=settings.openai_base_url or None` line is the same bridge the prompt-versioning module walked: when `OPENAI_BASE_URL` is empty (direct OpenAI), the SDK uses its default; when it is set to the Vocareum URL, the SDK routes through the proxy. Same code, two deploy targets, no conditional. The exercise inherits this without any new configuration. The return tuple adds `latency_ms` next to `(answer, TokenUsage, cost_usd)` — `src.generator.generate` returns only the first three because the cost-monitoring module's instrumentation did not need wall-clock latency, but A/B analysis does. Exercise 3 reads `latency_ms` for the per-variant latency aggregate.

The two templates live next to the base prompt at `prompts/`. `docbot_system_A.j2` is a verbatim copy of `docbot_system.j2`. `docbot_system_B.j2` differs in exactly one instruction — number 4 changes from "Be concise and direct" to "Be expansive" with a request to add a sentence or two of related context from the documentation. The concept module named multi-variable variant drift as a source of "the two variants are different in seven ways and you cannot attribute the metric shift to any one of them"; constraining the diff to one instruction is the experimental-design discipline. The `tests/test_ab.py::test_call_with_variant_renders_correct_template` test pins this by checking that "Be expansive" appears in variant B's system prompt and not in variant A's.

`log_assignment` at `src/optimization/ab.py:162-211` writes one JSONL row per call to a path the analyzer reads. The schema is small and stable: `client_id`, `variant`, `question` (truncated to 500 chars), `answer` (truncated to 2000), `latency_ms`, `prompt_tokens`, `completion_tokens`, `cost_usd`, and `success`. The `client_id` field is preserved as null when the per-request fallback ran, so the analyzer can distinguish sticky rows from fallback rows after the fact — useful when you want to compare the two assignment schemes on the same dataset.

## Part 3 — Wiring through the gateway and the honest seam

The gateway module added the `X-Client-Id` header to the gateway's `POST /query` route. Read `src/gateway/routes.py:48-62`:

```python
@router.post(constants.QUERY_ROUTE, response_model=QueryResponse)
def query_endpoint(
    request: QueryRequest,
    client_id: Annotated[
        str | None, Header(alias=constants.CLIENT_ID_HEADER)
    ] = None,
) -> QueryResponse:
    return route_query(
        request.question,
        top_k=request.top_k,
        client_id=client_id,
    )
```

The header value flows to `route_query(..., client_id=client_id)` at `src/gateway/router.py:38-76`. Today `route_query` accepts the keyword and forwards it for downstream use without consuming it — that is the contract the gateway module shipped so this module had a clean place to hook in. To turn on A/B for the gateway, you wrap `route_query`:

```python
def ab_route_query(question, *, client_id, traffic_split, salt):
    variant = pick_variant(client_id, traffic_split, salt=salt)
    # ... retrieve sources via the same pipeline route_query uses ...
    return call_with_variant(question, sources, variant)
```

The honest seam: this demo does not modify `src/gateway/router.py` to permanently swap `route_query` for `ab_route_query`. Exercise 1 wraps it from a script instead, which keeps the cache, tracing, cost-logging, and classifier layers `route_query` already composes from being disturbed. In a production retrofit you would push the variant selection into `route_query` itself, log the variant onto the trace span (the observability module's pattern), and let the existing observability flow carry the A/B labels downstream. That refactor is on the order of a day's work and out of scope for a thirty-minute module; the exercise's harness is the right shape for a development-time experiment.

Run the demo once from the project root with a tiny `client_id` pool to confirm the wiring:

```python
from src.models import Source
from src.optimization import pick_variant, call_with_variant
sources = [Source(doc_id="d1", chunk_text="DBSCAN clusters by density.", similarity_score=0.95)]
for cid in ["alice", "bob", "carol", "alice"]:  # alice appears twice
    v = pick_variant(cid, {"A": 0.5, "B": 0.5}, salt="prompt-style-v1")
    print(cid, v)
```

Four lines, alice's two assignments identical, bob and carol independent. That is sticky-by-user. To close, name what this demo addresses from the concept module's four LLM-specific pitfalls and what it does not. Cost asymmetry is handled — `log_assignment` carries `cost_usd` per call, so Exercise 3's analyzer can break it down by variant and surface a price-per-call difference that a chi-squared on success rate alone would miss. Sticky-by-user assignment is the deliberate design — the gateway header, the hash function, and the salt all exist for it. Per-request assignment is the documented fallback for the capstone's user-less workload. Judge variance is out of scope because the success metric in Exercise 1 is a citation check, not a judge score — when you graduate to a judge, the evaluation module's calibration discipline applies. Distribution drift would need wall-clock time the simulation does not consume. Two pitfalls handled, one named-and-fallback, two named-and-deferred. Bring the same discipline forward when you ship a real A/B harness behind the gateway.

---

# Exercises — Run a Sticky-by-User Simulation, Test for Significance, Decide on a Winner

Three exercises. The first wires the demo's `pick_variant` and `call_with_variant` into a runnable harness, fires two hundred calls across a fifty-`client_id` pool, and asserts that the same `client_id` always lands on the same variant — the sticky-ness verification. The second runs `make ab-analyze` to read the resulting JSONL log into a two-by-two contingency table, runs `scipy.stats.chi2_contingency`, and asks you to read the p-value honestly with the sticky-specific wrinkle that the effective sample size is the unique-client count, not the call count. The third adds cost and latency comparisons and forces a written decision; the engineering judgment is what closes the loop. Each exercise has a `Success Criteria` block that names what "done" looks like. Common pitfalls are at the end and worth reading before you start; the cost asymmetry warning in particular has caught learners on prior cohorts.

You will write one new file at `scripts/ab_simulate.py` (the harness — the active-learning bit). The starter already ships `scripts/ab_analyze.py`, the two prompt variant files at `prompts/docbot_system_A.j2` and `_B.j2`, the three primitives in `src/optimization/ab.py`, and the `make ab-analyze` Makefile target. Plan for roughly twenty minutes total, weighted toward Exercise 2 where the analysis happens.

## Setup

From the project root, you should have `make verify` passing and `make load-data` reporting the scikit-learn corpus is loaded. `.env` carries `OPENAI_API_KEY` and (on Vocareum) `OPENAI_BASE_URL` set to `https://openai.vocareum.com/v1`. If you are unsure which environment you are on, run:

```
uv run python -c "from src.config import settings; print(repr(settings.openai_base_url))"
```

You want `''` on direct OpenAI or `'https://openai.vocareum.com/v1'` on Vocareum. Any other value will fail in confusing ways downstream; fix the `.env` before continuing.

The exercise script will write to `data/ab_log.jsonl`. The starter's own cost log lives at `data/cost_log.jsonl` (added by the cost-monitoring module); the A/B log is a separate stream so the two do not interleave during the exercise. You can delete `data/ab_log.jsonl` between runs without affecting anything else in the starter.

You do not need `make serve` running. The exercise calls OpenAI directly through `src.config.settings` rather than going through the FastAPI gateway. That keeps the harness independent of the cache, the guardrails, and the cost-tracking middleware so the variant-versus-variant comparison is uncontaminated by other layers. The demo already showed that wiring `pick_variant` into the live gateway is a one-day refactor; the exercise's standalone harness is the right shape for a development-time experiment.

## Exercise 1 — Write the harness and run 200 simulated calls

Two hundred calls across fifty `client_id` values, with `pick_variant` doing the sticky-by-user routing and `log_assignment` recording one JSONL row per call. The point is to confirm the routing is sticky (the same `client_id` always lands on the same variant) and to produce a real log the next two exercises will analyze.

### What to do

1. Confirm the variant prompts are in place. From the project root:

   ```
   diff prompts/docbot_system_A.j2 prompts/docbot_system_B.j2
   ```

   You should see exactly the instruction-4 substitution and nothing else. Variant A says "Be concise and direct"; variant B says "Be expansive." The concept module named multi-variable variant drift as a source of confounded results — constraining the diff to one instruction is the experimental-design discipline this enforces.

2. Write `scripts/ab_simulate.py`. The structure is:

   ```python
   """200-call sticky-by-user A/B harness for the ScikitDocs assistant.

   Builds a 50-client_id pool, picks each call's client_id at random
   from the pool, calls pick_variant with a stable salt so assignments
   are sticky across calls, calls OpenAI through call_with_variant, and
   appends one JSONL row per call to data/ab_log.jsonl. The analyzer
   reads that file.
   """
   import random
   from pathlib import Path

   from src.models import Source
   from src.optimization import call_with_variant, log_assignment, pick_variant
   from src.pipeline import run_pipeline  # for real retrieval

   N_CALLS = 200
   N_CLIENTS = 50
   TRAFFIC_SPLIT = {"A": 0.5, "B": 0.5}
   SALT = "prompt-style-v1"
   LOG_PATH = Path("data/ab_log.jsonl")

   QUESTIONS = [
       "What is the default criterion for RandomForestClassifier?",
       "How does HistGradientBoostingRegressor handle missing values?",
       "What are the supported solvers for LogisticRegression?",
       "Does DBSCAN require the number of clusters as input?",
       "What's the difference between fit_transform and transform?",
       # ... add five to ten more from your retrieval set
   ]

   def retrieve(question):
       # Reuse the starter's pipeline retrieval seam — src/pipeline.py
       # exposes run_pipeline which returns a QueryResponse with .sources.
       # For the exercise you can short-circuit and call src.store.query
       # directly, or build a small in-memory fixture of Source objects.
       resp = run_pipeline(question, top_k=5)
       return resp.sources

   def main():
       LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
       clients = [f"user-{i:03d}" for i in range(N_CLIENTS)]
       for i in range(N_CALLS):
           question = random.choice(QUESTIONS)
           client_id = random.choice(clients)
           sources = retrieve(question)
           variant = pick_variant(client_id, TRAFFIC_SPLIT, salt=SALT)
           answer, usage, cost, latency_ms = call_with_variant(
               question, sources, variant
           )
           source_ids = {s.doc_id for s in sources}
           success = any(sid in answer for sid in source_ids)
           log_assignment(
               LOG_PATH,
               client_id=client_id,
               variant=variant,
               question=question,
               answer=answer,
               usage=usage,
               cost_usd=cost,
               latency_ms=latency_ms,
               success=success,
           )
           if (i + 1) % 20 == 0:
               print(f"{i+1}/{N_CALLS} done")

   if __name__ == "__main__":
       main()
   ```

3. Run it from the project root:

   ```
   uv run python scripts/ab_simulate.py
   ```

   Expect the run to take roughly four to six minutes on Vocareum (the proxy serializes calls under load) or roughly two to three minutes on direct OpenAI. The `(i+1)/200 done` progress print at every twentieth call tells you the script is alive; the cold-start latency on the first call is the OpenAI client warmup and is normal.

4. Verify stickiness directly. Run this one-liner from the project root:

   ```
   uv run python -c "import json; from collections import defaultdict; \
       by_client = defaultdict(set); \
       [by_client[r['client_id']].add(r['variant']) for r in (json.loads(l) for l in open('data/ab_log.jsonl'))]; \
       print('multi-variant clients:', sum(1 for v in by_client.values() if len(v) > 1))"
   ```

   The expected output is `multi-variant clients: 0`. Every `client_id` should appear with exactly one variant. If you see a non-zero number, the harness is calling `pick_variant` without a `client_id` or with different salts across calls — the sticky contract is broken and Exercises 2 and 3 will report compromised numbers.

### Cost asymmetry warning

Two hundred calls at `gpt-4o-mini` on this workload runs roughly two to five cents total, hedged because the per-call cost depends on how long your retrieval context is and how verbose variant B's responses come out. If you accidentally point the model at `gpt-4o` instead — `call_with_variant` accepts a `model` kwarg — that becomes roughly twenty to forty cents, about an order of magnitude more, and the difference compounds across iterations. The pricing concept module's rate table is the authoritative reference; the concept module named cost asymmetry as the first LLM-specific A/B pitfall and this is the per-call reason why. Keep the model on `gpt-4o-mini` (the default) for the exercise and only bump up if you are deliberately exploring how the conclusion changes with a more expensive model.

The reason cost asymmetry matters here, not just in the abstract, is that variant B asks for longer answers. Output tokens price at four times input tokens on `gpt-4o-mini`, so any prompt change that doubles the response length roughly doubles the per-call cost. In production, that would translate to a doubled monthly bill at constant traffic volume, and the chi-squared test on success rate alone would not surface the cost gap. The cost-monitoring module's dashboard at `src/cost/dashboard.py` is the production layer for catching this; the JSONL log here is the development-time analog you can reason about with a one-liner instead of a dashboard.

### Success Criteria

- `prompts/docbot_system_A.j2` and `docbot_system_B.j2` differ only on instruction 4 (`diff` shows one substitution).
- `scripts/ab_simulate.py` runs to completion without exceptions and produces `data/ab_log.jsonl` with roughly two hundred lines.
- Each line parses as JSON and contains the nine fields the schema defines (`client_id`, `variant`, `question`, `answer`, `latency_ms`, `prompt_tokens`, `completion_tokens`, `cost_usd`, `success`).
- Every `client_id` appears with exactly one variant — the multi-variant-clients check returns zero.
- The split between variant A and variant B is roughly one hundred each, within reasonable random-sampling jitter (anything in `[80, 120]` per variant is fine at this N — your client pool's hash distribution will jitter the split slightly).
- Total cost from `data/ab_log.jsonl` is on the order of a few cents, not a few dollars.

### Stretch

Add a `--salt` CLI argument and run the simulation twice with different salt values against the same client pool. Concatenate the two log files and rerun the stickiness check — within each salt, the multi-variant-clients count should still be zero. Across salts, roughly half of the client_ids should flip variants (this is the salt-isolation property the `test_pick_variant_salt_isolates_experiments` test pins on the unit-test side). Naming what the salt controls and what it does not is the discipline.

## Exercise 2 — Analyze with `make ab-analyze` and read the result honestly

The starter ships `scripts/ab_analyze.py` and a `make ab-analyze` target that reads `data/ab_log.jsonl`, builds the variant × success contingency table, runs `scipy.stats.chi2_contingency`, and prints the per-variant aggregates Exercise 3 will read. The interpretation is where most of the learning lives — at two hundred sticky calls across fifty clients the test will almost certainly come back insignificant, and the correct read is "we do not have enough effective data," not "the variants are equivalent."

### What to do

1. Run the analyzer from the project root:

   ```
   make ab-analyze
   ```

   You will see something like:

   ```
   Loaded 200 rows from the A/B log.
   Unique client_ids (sticky effective N): 50

   Variant A: 87/103 success (84.5%)
   Variant B: 81/97 success (83.5%)
   chi2 statistic:                0.038
   p-value:                       0.8451
   degrees of freedom:            1
   significant at alpha=0.05:     False
   ```

2. Read the p-value honestly. A p-value of roughly 0.85 means: even if the two variants produced identical success rates in the long run, a one-percentage-point gap in either direction at N=200 happens by chance about eighty-five percent of the time. The honest read is "we cannot distinguish these variants at this sample size." That is also the result you would expect for any two variants that differ by less than about ten percentage points in success rate when each variant has roughly one hundred observations. The chi-squared test does not say "the variants are the same"; it says "the data does not give us enough evidence to claim they differ." The distinction matters because "no evidence of a difference" and "evidence of no difference" are different statistical claims — the second one requires an equivalence test, far more samples, and a pre-declared equivalence margin.

3. Read the sticky-specific wrinkle. The analyzer's first line prints `Unique client_ids (sticky effective N): 50`. That is the number that matters for power, not the raw call count of two hundred. Sticky-by-user assignment correlates the four calls each user contributes — if user-007 lands on variant A, all four of their success outcomes are draws from variant-A's distribution conditioned on user-007's behavior, not four independent draws from variant A overall. Kohavi, Tang, and Xu's 2020 chapter on clustered randomization covers the formal version; the intuition is that the effective sample size is closer to the unique-client count than the call count when within-user calls are highly correlated. So this test is more underpowered than the same N=200 run under per-request randomization would have been. That is the cost of sticky — you trade statistical power for user-experience coherence and the ability to measure user-level metrics that per-request cannot reach. The concept module named the tradeoff; Exercise 2 is where you feel it numerically.

4. Reason about sample size. For a small relative shift in a binary metric at alpha=0.05 and power=0.8 under independent observations, the textbook formula gives on the order of ten thousand to twenty thousand observations per variant. Sticky-by-user with high within-user correlation can push that number up by a factor of three to five — you might need 30k-50k unique clients per variant for a small effect. Larger effects need fewer; a fifteen-percentage-point absolute shift needs roughly two to three hundred per variant under independence, scaled accordingly for sticky. Fifty clients on each side puts you near the floor of detectable effects, and even a clean per-request run at N=200 would be similarly underpowered for typical prompt-variant effect sizes. Pre-declaring the minimum-meaningful effect size is what tells you whether to run two hundred calls, two thousand, or twenty thousand — and how many *unique* clients you need behind those calls.

### Success Criteria

- `make ab-analyze` runs end-to-end and prints the contingency table, chi-squared statistic, p-value, dof, and an `alpha=0.05` boolean.
- The unique-client count is printed and is roughly fifty (matching the pool size you chose in Exercise 1).
- Your written interpretation explicitly names whether the result is significant at the alpha=0.05 threshold AND addresses the sticky-effective-N wrinkle. "Underpowered for sticky-by-user at fifty unique clients, no detectable difference" is the most likely honest read and is perfectly acceptable.

### Stretch

Rerun `scripts/ab_simulate.py` with `N_CLIENTS = 500` and `N_CALLS = 1000` and report whether the conclusion changes. The effective sample size is now 500 unique clients with two calls each; the test should be roughly ten times more powerful than the N=50 run. If variant B really is meaningfully different from A — verbose answers may genuinely cite more source IDs because they contain more text — the larger sample should make the difference detectable. If the variants are statistically indistinguishable at N=1000, that itself is a useful result.

## Exercise 3 — Cost and latency comparison, plus a written decision

Quality parity does not settle the question. The concept module was explicit: two variants that score statistically tied on quality can still have meaningfully different cost or latency profiles, and either dimension can decide the winner. The analyzer already computes both; this exercise reads them and forces a written decision.

### What to do

1. Read the per-variant table the analyzer printed. The bottom block looks like:

   ```
   metric                             A           B
   --------------------------------------------------
   n                                103          97
   mean_latency_ms                  631         984
   p50_latency_ms                   604         942
   mean_cost_usd               0.00012     0.00018
   total_cost_usd              0.01236     0.01746
   mean_completion_tokens            38          67
   ```

2. Walk the example. Variant B's "be expansive" instruction roughly doubles the output token count, which roughly doubles the completion cost (output tokens price at four times input tokens on `gpt-4o-mini`, and prompt tokens are unchanged across variants because the variants share the same retrieval and only differ on one instruction). Latency tracks output length because the model streams more tokens. So at quality parity (Exercise 2's null result), variant A wins on cost and latency. The deciding metric is not quality — it is throughput. The general pattern this surfaces: when two prompt variants are statistically indistinguishable on the primary metric you started measuring, the secondary metrics decide the winner. In a customer-support workload, latency matters because slow answers feel worse than fast ones at the same quality. In a high-volume documentation Q&A workload like ScikitDocs, cost matters because the bill scales linearly with traffic and a fifty-percent cost gap is a fifty-percent budget gap. The concept module named these as "guardrail metrics" — the ones that should not move the wrong way even if the primary moves the right way — and Exercise 3 is the practical application.

3. Write a decision docstring at the top of `scripts/ab_simulate.py`. Three or four lines, formatted like:

   ```
   """A/B decision (2026-05-19): Variant A retained.

   Chi-squared on success rate at 50 unique clients returned p=0.85
   (not significant; effective N is sticky-correlated so this is even
   more underpowered than the raw call count suggests). Variant B was
   ~50% more expensive per call and ~56% slower in mean latency with
   no detectable quality improvement. Next step: rerun at 500 unique
   clients to confirm the quality-parity read holds at higher power,
   then sunset variant B unless a quality signal emerges.
   """
   ```

   The format matters less than the content. The decision should name the winning variant, the metric the decision rests on, the confidence (or lack of it) in the read, and the proposed next step. If the result is truly underpowered, "we need more data" is an acceptable decision — just write it down explicitly so the reader knows the test was not conclusive. Naming the sticky-effective-N caveat alongside the raw call count is what makes the write-up honest about the design choice's cost.

### Success Criteria

- The analyzer's per-variant block prints n, mean and median latency, mean and total cost, and mean completion tokens (verify by reading the `make ab-analyze` output).
- The decision docstring is present at the top of the simulation script, names a winning variant (or explicitly defers to "more data needed"), and grounds the decision in numbers from the run — not in vibes.
- If the chi-squared result is non-significant, the docstring says so AND names the sticky-effective-N wrinkle. If the cost or latency gap exceeds ten percent, the docstring names it as the deciding factor.

### Stretch

Replace the simple `success = any(sid in answer for sid in source_ids)` heuristic with an LLM-as-judge faithfulness score. Use `gpt-4o-mini` as the judge, feed it the question, the sources, and the answer, and ask it to return a JSON verdict like `{"supported": true/false, "reason": "..."}`. Run this against the existing log and compare the chi-squared result on the judge label to the citation-check label. The evaluation module's framing applies — the judge itself has variance, so a single judge call per response is the cheap version and the rigorous version averages three calls. If the conclusion flips between the two labels, the underlying signal is weaker than either label suggests.

## Common Pitfalls

A few traps that catch most learners on this module. Skim before submitting.

- **Forgetting to pass `client_id` to `pick_variant`.** Without it, the function silently falls back to per-request weighted random sampling, your `data/ab_log.jsonl` rows show `client_id: null`, and the stickiness check finds multi-variant clients (because re-rolls happen per call). The fallback is deliberate so the harness doesn't raise on un-headered traffic; the assertion in Exercise 1 step 4 is the canary.
- **Forgetting to pin the salt across calls.** If you let the salt vary, each call hashes to an independent variant for the same `client_id` — also broken sticky-ness. Keep `SALT` as a module-level constant or a CLI arg, never a per-call random.
- **Reading `cost_usd` as a string from JSONL.** `json.loads` returns the float correctly, but if you `pandas.read_csv` the file or do any string-based manipulation, you can end up summing strings concatenated with `+` and getting nonsense. The analyzer reads each line through `json.loads`, which yields the right types directly.
- **Variants differ in more than one variable.** If you change instruction 4 and also reword instruction 2, the chi-squared result is contaminated by both changes and you cannot attribute a metric shift to either one. Run `diff prompts/docbot_system_A.j2 prompts/docbot_system_B.j2` before you start and confirm it shows exactly the one substitution.
- **Vocareum proxy rate-limit timeouts on a long loop.** Two hundred calls in a tight loop can hit the Vocareum proxy's per-key throttle. If you see `429` or socket timeouts, add `time.sleep(0.5)` between calls or run during off-peak hours. The starter's eval Makefile target defaults to single-worker concurrency for the same reason.
- **Pointing the script at `gpt-4o` instead of `gpt-4o-mini`.** Costs go up roughly tenfold on the same workload. The asymmetry warning at the end of Exercise 1 is the canonical example of why the concept module named cost asymmetry as the first LLM-specific A/B pitfall.
- **Interpreting a non-significant p-value as "the variants are equivalent."** That is not what the test says. It says "the data does not let us reject the null hypothesis of equivalence at the chosen alpha." The two are different — equivalence requires a different test family (equivalence tests, two one-sided tests) and far more data than this exercise produces. State the underpowered finding honestly.
- **Confusing the raw call count with the sticky effective N.** Two hundred calls across fifty clients is *not* the same as two hundred independent observations. The analyzer prints both numbers; the effective N is what powers the test, and the concept module's framing on within-user correlation is the conceptual handle. The capstone's per-request workload would not have this issue — it would have the opposite issue (no user-level metrics available at all). Naming which workload you are on and which scheme it implies is the discipline.

What you have at the end. A two-hundred-call A/B harness that pushes traffic through two prompt variants via a sticky-by-user feature flag in `src/optimization/ab.py`, a chi-squared analyzer that reads the result honestly even when the result is "underpowered for the sticky design," and a per-variant cost and latency comparison plus a written decision that grounds the next step in numbers. In production you would push the flag into OpenFeature with a LaunchDarkly, Statsig, or Flipt backend; you would replace the citation-check success metric with a judge score or a behavioral signal; and you would run the test long enough to satisfy the sample-size formula Kohavi, Tang, and Xu derived in 2020 — corrected for clustering. Each of those upgrades is a layer on top of what you built here; the routing, logging, and analysis skeleton stays the same.
