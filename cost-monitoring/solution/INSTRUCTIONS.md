# Module 13 — Cost Monitoring with `compute_cost`, JSONL Logging, and a Live Dashboard

## Setup

This starter is the ScikitDocs RAG app — a Q&A assistant for the scikit-learn library — with the full instrumentation stack already wired: prompt loader and Jinja templates, Chroma vector store, RAG pipeline (`run_pipeline`), Phoenix tracing, RAGAS evaluation harness, cost monitoring (`src/pricing.py` + `src/cost/`), semantic answer cache, FastAPI gateway, guardrails, A/B testing, RAGOps watcher, and a streaming endpoint. In this module you will exercise the cost stack end-to-end: seed a 50-row synthetic log, fire real queries and watch the JSONL grow, then author a rolling-baseline anomaly alert plus a pre-call `tiktoken` budget gate.

Bring up the corpus before you start:

```bash
uv sync
cp .env.example .env          # add your OPENAI_API_KEY (or Vocareum voc- key)
make load-data                # ~45–60s cold, ~5s warm; ~$0.10 in embeddings
make seed-difficulty          # upsert 8 deliberately-confusing chunks
```

Smoke-check the pipeline before any cost work:

```bash
uv run python -c "from src.pipeline import run_pipeline; print(run_pipeline('What kernel does SVC use by default?').answer[:80])"
```

If that returns a grounded answer about `rbf`, you are ready. Follow the demo walkthrough first, then work through the three exercises in order.

---

# Exercises — Seed the Log, Instrument Real Queries, Build an Alert and a Budget Gate

The demo wired `compute_cost` into `src/generator.py`, fired one query through `run_pipeline`, watched a row land in `data/cost_log.jsonl`, and rendered the dashboard at `localhost:8080/cost-dashboard`. These exercises put the workflow in your hands. Three exercises, three artifacts: a fifty-row seeded log with a per-day breakdown and a costliest-query-type callout, a real twenty-query instrumented run with a cost delta, and a working rolling-baseline alert plus a pre-call tiktoken budget gate. Plan for twenty minutes, weighted slightly toward Exercise 3 because the muscle memory for both alerting and budget-gating is what production cost work pays for.

## Setup notes specific to the exercises

Same as the demo, with `make load-data` reporting the corpus is in Chroma and your `.env` carrying `OPENAI_API_KEY` plus `OPENAI_BASE_URL` if you are on Vocareum. Exercise 1 needs no live LLM calls because `make seed-cost-log` populates the log synthetically. Exercises 2 and 3 do hit the API — budget a few cents on Vocareum or any other endpoint; on `gpt-4o-mini` at sixty cents per million output tokens, twenty queries at the starter's median completion length runs well under five cents in total.

A sanity check before you start: `cat data/cost_log.jsonl 2>/dev/null | wc -l` confirms whether you already have rows. If the count is zero or the file is missing, the seed in Exercise 1 starts from a clean slate. If the count is non-zero from prior demo runs, your choice is to delete the file or pass `--reset` to the seed script — the script default is idempotent (it appends until it hits the target and skips if the target is already met).

## Exercise 1 — Seed fifty entries, produce a per-day breakdown, name the costliest query type

A useful operational target is fifty entries in `data/cost_log.jsonl` — enough rows that per-day and per-model rollups have something to chew on. The starter ships `make seed-cost-log` so you can hit that floor in under a second instead of waiting through a fifty-query bash loop and paying for the calls. This exercise uses the seed, then runs a small aggregation against it to produce a per-day breakdown and identify the costliest day and costliest query type. The output is an artifact you paste into your writeup.

### What to do

1. Seed the log:

   ```bash
   make seed-cost-log
   ```

   The target resolves to `uv run python scripts/seed_cost_log.py`. The script is deliberate about the mix: 70% `simple` (`gpt-4o-mini`), 20% `complex` (`gpt-4o`), 10% `hallucination_check` (`gpt-4o-mini`). Token-count distributions are eyeballed from a real ScikitDocs run — prompts cluster around 1,100 to 1,300 tokens, completions vary by query type. Timestamps spread evenly across the past 24 hours so the per-day rollup has something to chew on. The deterministic seed (`SEED = 20260501`) means re-running with `--reset` produces the exact same fifty rows, which is the right behavior for reproducible grading.

   On a clean log you should see:

   ```
   seeded 50 entries (0 existing + 50 new = 50 total)
   ```

   Confirm the row count:

   ```bash
   wc -l data/cost_log.jsonl
   ```

   You want fifty.

2. Run a baseline cost summary. The starter does not ship a separate `cost-report` target — the renderer in `src/cost/dashboard.py` already shows totals plus per-model breakdown when you hit `make cost-dashboard`, and the same numbers fall out of `summarize()` in five lines:

   ```bash
   uv run python -c "
   from src.cost.tracker import load_log, summarize
   s = summarize(load_log())
   print(f'Total requests: {s[\"total_requests\"]}')
   print(f'Total cost:     \${s[\"total_cost_usd\"]:.4f}')
   for model, stats in sorted(s['by_model'].items()):
       print(f'  {model}: {stats[\"requests\"]} reqs, \${stats[\"cost_usd\"]:.4f} (\${stats[\"avg_cost_usd\"]:.6f} avg)')
   "
   ```

   On a freshly seeded log this prints something like:

   ```
   Total requests: 50
   Total cost:     $0.0722
   gpt-4o: 10 reqs, $0.0642 ($0.006422 avg)
   gpt-4o-mini: 40 reqs, $0.0080 ($0.000199 avg)
   ```

   Copy these lines into your writeup. Notice the asymmetry — `gpt-4o` is 20% of the requests but ~89% of the dollar total. That ratio is the asymmetric-pricing pattern the concept module anchored, made concrete on your local log.

3. Produce the per-day breakdown. The starter does not ship a per-day rollup script either, so write a one-liner:

   ```bash
   uv run python -c "
   import json
   from collections import defaultdict
   by_day = defaultdict(lambda: {'cost': 0.0, 'requests': 0})
   for line in open('data/cost_log.jsonl'):
       r = json.loads(line)
       day = r['timestamp'][:10]
       by_day[day]['cost'] += r['cost_usd']
       by_day[day]['requests'] += 1
   for day, stats in sorted(by_day.items()):
       print(f'{day}: \${stats[\"cost\"]:.4f}  ({stats[\"requests\"]} requests)')
   "
   ```

   That prints one line per day. Because the seed spreads across 24 hours from the moment you ran it, you should see two days — yesterday and today — and the split depends on what hour you ran the seed. The costliest day is the one that happened to land more `complex` rows (the `gpt-4o` calls dominate the dollar total even at 20% of the request count).

4. Identify the highest-cost query type. The same shape, grouped on `query_type`:

   ```bash
   uv run python -c "
   import json
   from collections import defaultdict
   by_type = defaultdict(lambda: {'cost': 0.0, 'requests': 0})
   for line in open('data/cost_log.jsonl'):
       r = json.loads(line)
       by_type[r['query_type']]['cost'] += r['cost_usd']
       by_type[r['query_type']]['requests'] += 1
   for qt, stats in sorted(by_type.items(), key=lambda x: -x[1]['cost']):
       avg = stats['cost'] / stats['requests']
       print(f'{qt}: \${stats[\"cost\"]:.4f}  ({stats[\"requests\"]} requests, avg \${avg:.6f})')
   "
   ```

   On the seeded mix you will see `complex` at the top of the dollar total even though it is only 20% of requests — that is the asymmetric-pricing pattern again. Average cost per `complex` request is roughly twenty to forty times the average cost per `simple` request because both the model rate is ten-plus times higher (gpt-4o costs ~17× gpt-4o-mini on input, ~17× on output) and the average completion length is roughly five to six times longer. Compose those ratios and you get a per-request cost ratio that compounds.

### Acceptance criterion

Three artifacts pasted into your writeup. The total + per-model breakdown lines from step 2. The per-day breakdown table. The per-query-type breakdown ranked by dollar total with averages. A one-paragraph note naming the costliest day, the costliest query type, and the ratio between `complex` and `simple` average cost per request. A teammate reading the writeup should be able to trace every number back to a row in `data/cost_log.jsonl`.

## Exercise 2 — Instrument twenty real ScikitDocs queries and watch the log grow

Exercise 1 used synthetic data. Exercise 2 fires real queries through `run_pipeline` and confirms the cost log captures them with the same shape the seed produces. This proves the instrumentation works end-to-end — usage object off the OpenAI response, `compute_cost` against the model rate table, one JSONL row appended per call — and produces the cost delta a real (small) load adds. A gateway can call `log_request` automatically on every `/query`; for this exercise you call both functions by hand so you can see the contract.

### What to do

1. Decide whether to keep the seeded log or start fresh. If you want this exercise's twenty rows isolated:

   ```bash
   mv data/cost_log.jsonl data/cost_log.seed.jsonl
   ```

   Or accept that this run appends to the existing fifty rows. Both are valid — the writeup just needs to name which path you took.

2. Write a small Python script that fires twenty queries with mixed shape. Use five scikit-learn questions from `data/golden_set.csv` covering different `query_type` buckets so the model routing (here you tag manually) has variety:

   ```bash
   uv run python -c "
   from src.pipeline import run_pipeline
   from src.cost.tracker import log_request

   questions = [
       ('What kernel does \`SVC\` use by default?', 'simple'),
       ('What is the default value of \`n_clusters\` in \`KMeans\`?', 'simple'),
       ('What does \`StandardScaler\` do to input features?', 'simple'),
       ('Compare \`RandomForestClassifier\` and \`GradientBoostingClassifier\` for tabular tasks.', 'complex'),
       ('Recommend a pipeline for high-cardinality categorical features with a tree model.', 'complex'),
   ]
   for q, qt in questions:
       for i in range(4):
           r = run_pipeline(q)
           log_request(r.model, r.tokens, r.cost_usd, query_type=qt)
   "
   ```

   That fires twenty queries (five distinct questions, four repetitions each). The repetition is deliberate for the bonus: OpenAI's auto-caching activates on identical prefixes above 1,024 tokens, and the starter's `docbot_system.j2` plus five retrieved chunks comfortably exceeds that floor, so the second and later repetitions of each question may hit the prompt cache and bring `cached_tokens` back on the usage object.

3. Confirm the log grew by twenty. If you renamed the seed file:

   ```bash
   wc -l data/cost_log.jsonl
   ```

   should report twenty. If you appended to the seed, the count should be seventy.

4. Re-run the cost summary against the post-load log:

   ```bash
   uv run python -c "
   from src.cost.tracker import load_log, summarize
   s = summarize(load_log())
   print(f'Total: {s[\"total_requests\"]} requests, \${s[\"total_cost_usd\"]:.4f}')
   for model, stats in sorted(s['by_model'].items()):
       print(f'  {model}: \${stats[\"cost_usd\"]:.4f} ({stats[\"avg_cost_usd\"]:.6f} avg)')
   "
   ```

   The dollar totals are now real money (well, real Vocareum quota). Compare against the seeded baseline: total cost climbs by what your twenty rows actually cost. For your writeup, this is the post-load artifact.

5. **Bonus — prompt caching.** Inspect the last twenty rows for `prompt_tokens` patterns:

   ```bash
   tail -20 data/cost_log.jsonl | uv run python -m json.tool --json-lines | grep prompt_tokens
   ```

   Repeated questions through the same system prompt produce nearly identical `prompt_tokens` counts across runs — the variability is whichever retrieved chunks land on top, which is small for a stable corpus. The starter's `TokenUsage` at `src/models.py:20-28` does not yet expose OpenAI's `prompt_tokens_details.cached_tokens` field, so the cost log under-resolves cache reads as plain input. The architectural fix is two-step: extend `TokenUsage` with an optional `cached_tokens` field, then teach `compute_cost` to apply the cached-input rate (one dollar twenty-five per million for `gpt-4o`, a 50% discount on the cached portion per the OpenAI pricing page as of May 2026; for `gpt-4o-mini` the cached rate is seven-and-a-half cents per million, also 50% off). For this exercise the writeup deliverable is the two-step plan plus a back-of-envelope estimate of what your twenty-query load would have cost with cache-aware pricing. Use the fraction of repeated prefixes (fifteen of twenty for this loop — four repeats of each of five questions, three are repeats) times the prefix size times the 50% discount to estimate the unrealized savings. On a typical run that lands in the half-cent to one-cent range for twenty queries; the percentage is what matters, and on a workload with stable system prompts it lands in the 15-to-30% input-cost reduction band.

### Acceptance criterion

The new summary output captured in the writeup. A `tail -5 data/cost_log.jsonl` excerpt showing the entry shape on real traffic. A two-sentence note confirming the log grew by exactly twenty rows. For the bonus: a one-paragraph estimate of cache-aware savings on your twenty-query load, with the two-step extension plan to `TokenUsage` and `compute_cost`.

## Exercise 3 — Build an alert plus a pre-call tiktoken budget gate

The concept module named the two alerting patterns worth implementing: budget thresholds against a monthly cap, and anomaly detection against a rolling baseline. This exercise builds the rolling-baseline anomaly check and, as a second piece, a pre-call budget gate that uses tiktoken to estimate cost before the request reaches the API. Both artifacts are small — under fifty lines each — but they cover the operational surface that turns instrumentation into protection.

### What to do — Part A: rolling-baseline anomaly alert

1. Write a script `scripts/cost_alert.py` (or paste this into a one-liner if you prefer) that reads `data/cost_log.jsonl`, computes today's cost, computes the rolling seven-day average excluding today, and prints a warning if today's cost is more than twice the rolling average:

   ```python
   import json
   from collections import defaultdict
   from datetime import datetime, timedelta, timezone

   by_day = defaultdict(float)
   for line in open("data/cost_log.jsonl"):
       r = json.loads(line)
       day = r["timestamp"][:10]
       by_day[day] += r["cost_usd"]

   today = datetime.now(timezone.utc).date().isoformat()
   today_cost = by_day.get(today, 0.0)
   prior_days = [
       (datetime.fromisoformat(today) - timedelta(days=i)).date().isoformat()
       for i in range(1, 8)
   ]
   prior_costs = [by_day.get(d, 0.0) for d in prior_days]
   prior_costs = [c for c in prior_costs if c > 0]
   rolling_avg = sum(prior_costs) / len(prior_costs) if prior_costs else 0.0

   print(f"Today ({today}): ${today_cost:.4f}")
   print(f"Rolling 7d avg (excl today): ${rolling_avg:.4f}")
   if rolling_avg > 0 and today_cost > 2 * rolling_avg:
       print(f"WARNING: today's cost is {today_cost / rolling_avg:.1f}x the rolling average")
   ```

   Two things to notice. The script filters out zero-cost days from the rolling average — a brand-new log with seven days of history but only one day of activity should not divide by seven; it should divide by one. And the threshold is a multiple (two times), not an absolute dollar value, because the absolute number scales with traffic and the multiple is what catches a regression regardless of base rate. The Pluralsight "Meter Before You Manage" piece from February 2026 names this pattern directly: the cost-per-request spike is the canary, and the rolling-baseline multiple is how you operationalize it.

2. Run the script against your seeded plus real-traffic log:

   ```bash
   uv run python scripts/cost_alert.py
   ```

   On a seeded log spread across 24 hours, today and yesterday split roughly evenly; the rolling average is one number, today's is another, and the alert does not fire because the ratio is near one. To verify the alert fires when it should, run another twenty queries (extend the loop from Exercise 2) and re-run the alert — today's number is now meaningfully higher than the rolling average, and the warning prints.

### What to do — Part B: pre-call tiktoken budget gate

3. Write a function `estimate_cost(question, model, system_prompt_tokens=1200, expected_output_tokens=200)` that uses tiktoken to count the question's tokens, adds the system-prompt overhead, and returns the estimated USD cost from the rate table:

   ```python
   import tiktoken
   from src.pricing import MODEL_PRICING

   def estimate_cost(
       question: str,
       model: str,
       system_prompt_tokens: int = 1200,
       expected_output_tokens: int = 200,
   ) -> float:
       enc = tiktoken.encoding_for_model(model)
       question_tokens = len(enc.encode(question))
       input_tokens = question_tokens + system_prompt_tokens
       input_rate, output_rate = MODEL_PRICING[model]
       return (
           input_tokens * input_rate
           + expected_output_tokens * output_rate
       ) / 1_000_000
   ```

   Two design notes. `encoding_for_model("gpt-4o-mini")` returns the `o200k_base` encoding; `encoding_for_model("gpt-3.5-turbo")` returns `cl100k_base`. Using the wrong encoding miscounts by a few percent on typical English prose and considerably more on code or non-English text — the common pitfall section names this as the failure mode to watch. The `expected_output_tokens` parameter is a guess; production teams calibrate it from the median completion length per query type in their cost log. For the gate, what matters is that you have a number to put in the formula; the gate either passes or fails before the call goes out.

4. Wrap it in a gate that refuses calls above a per-request dollar limit:

   ```python
   def gate(question: str, model: str, limit_usd: float = 0.01) -> None:
       est = estimate_cost(question, model)
       if est > limit_usd:
           raise ValueError(
               f"Estimated cost ${est:.4f} exceeds limit ${limit_usd:.4f}"
           )
   ```

   Fire it against a normal question — `gate("What kernel does SVC use by default?", "gpt-4o-mini", 0.01)` returns silently because the estimated cost is well under a cent. Fire it against a pathologically long prompt with a tight limit — `gate("very long question..." * 1000, "gpt-4o", 0.001)` — and it raises before any tokens leave your machine.

5. Reconcile the estimate against the post-call truth. Fire the same normal question through `run_pipeline`, read `r.tokens.prompt_tokens` and `r.tokens.completion_tokens`, and compare `compute_cost(r.model, r.tokens)` against `estimate_cost(question, r.model)`. The estimate should land within roughly 10-30% of the truth — `prompt_tokens` is close (the docbot system prompt is stable, only the retrieved chunks vary) and `completion_tokens` is the wild card because output length is unknown until the model emits it. That gap is why the estimate is for budget gating, not for billing — the post-call `usage` object remains the ground truth for the cost log.

A representative reconciliation run on a ScikitDocs `simple` question lands around the following shape — your numbers will vary by a few percent because retrieved chunks differ in length, but the pattern is consistent across questions:

```
question: "What does StandardScaler do to input features?"
model:    gpt-4o-mini
estimate_cost (system=1200, expected_out=200): $0.000300
actual prompt_tokens:    1164
actual completion_tokens: 76
compute_cost(actual):    $0.000220
delta: estimate is 36% high (largely because expected_out=200 overestimated a 76-token answer)
```

That 36% high reading is the right kind of error for a budget gate — the gate refuses slightly too eagerly, never too late. Flip the same question on `gpt-4o` and the estimate-vs-truth gap narrows because the rate ratios cancel, and the dollar magnitude grows by 10×. Use the gate to refuse expensive calls before they leave your machine; use the cost log to account for what actually happened.

### Acceptance criterion

For Part A: a working `cost_alert.py` (or equivalent one-liner) and a captured output showing today's cost, the rolling average, and either the alert text or a one-sentence note that the alert did not fire and why. For Part B: a working `estimate_cost` function and `gate` function captured in your writeup, plus one reconciliation example — the estimated cost from tiktoken, the actual cost from the response, and the percentage delta. A one-paragraph note on which knob you would tune in production: the per-request limit, the `expected_output_tokens` guess, or both, and how you would calibrate from the cost log.

## Common pitfalls

A few traps that catch most learners on this module:

- **Tokenizer mismatch.** `tiktoken.encoding_for_model("gpt-5-something")` raises `KeyError` for any model name the library has not seen — the encoding tables update with each `tiktoken` release. Falling back to `tiktoken.get_encoding("o200k_base")` is the safe default for any GPT-4o-and-newer model; `cl100k_base` is the safe default for the older `gpt-3.5-turbo` and `gpt-4` generations. Using the wrong encoding does not crash, it silently miscounts — the budget gate then passes calls it should have refused, or refuses calls it should have passed.
- **`cached_tokens` undercounting.** Once OpenAI's auto-caching activates on a stable prefix, `response.usage.prompt_tokens` still reports the full input count, and `prompt_tokens_details.cached_tokens` reports how many of those were cache hits. The starter's current `compute_cost` treats every input token as full-rate, so the cost log overstates spend on cache-heavy workloads. The fix is the two-step extension named in Exercise 2's bonus.
- **Cost log file locking under concurrent writes.** `log_request` uses `open(path, "a")` and writes one line. On POSIX, append-mode writes shorter than `PIPE_BUF` (4 KiB on Linux) are atomic across processes — one JSONL row of cost data fits comfortably under that ceiling. On Windows or under multi-process writers where rows might exceed the buffer, the safe pattern is `os.O_APPEND | os.O_WRONLY` plus an explicit per-line flush, or a file lock. A single-process FastAPI server does not need either.
- **Stale model rates in `MODEL_PRICING`.** Vendor pricing pages move, and the starter's table is dated `as of 2026-04`. When you adopt this pattern for your own project, the first audit step is re-verifying the rates against the live pricing page on the day you ship. A stale table silently understates or overstates cost across every row.
- **Pre-call estimate vs post-call truth.** The estimate from tiktoken plus a heuristic `expected_output_tokens` lands within 10-30% of the post-call usage on most queries and is way off on the few queries where the model produces a much shorter or much longer answer than expected. That is fine for budget gating — the gate's job is to refuse pathological cases, not predict cost to the cent — and it is not fine for billing or chargeback. Use the gate for refusal; use the cost log for accounting.

## What you have now

A fifty-row seeded cost log with a per-day breakdown and a costliest-query-type callout. A twenty-query real-traffic load that proves the instrumentation works end-to-end on Vocareum or a direct OpenAI account. A rolling-baseline anomaly alert and a pre-call tiktoken budget gate, both as artifacts you can drop into a production codebase with minimal modification. Together those four artifacts cover the four-layer instrumentation stack: per-request log, aggregation, alerting, and the pre-call estimate that gates spend before it happens.

The instrumentation you wired today is upstream of every later cost decision. Semantic caching reduces cost by removing calls entirely — a cache hit returns the prior answer without a fresh model invocation, and your cost log records nothing for those requests. A gateway that mounts `/cost-dashboard` and calls `log_request` automatically on every `/query` makes the manual `log_request` calls from Exercise 2 implicit. A hallucination-check call added per response doubles the log volume on the protected endpoints, and Exercise 1's per-query-type breakdown is the diagnostic shape for that added cost.
