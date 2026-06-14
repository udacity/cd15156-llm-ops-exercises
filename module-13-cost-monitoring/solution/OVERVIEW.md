---
module_number: 13
module_title: "Cost Monitoring with compute_cost, JSONL Logging, and a Live Dashboard"
slug: cost-monitoring
unit_type: overview-video-script
target_minutes: "5-7"
word_count: 706
---

# Module 13 Overview Video: Cost Monitoring

> Recording aid for a 5-7 min overview (read pace ~100 wpm). Read all prose aloud, top to bottom. `*[Stage: ...]*` lines are screen directions, not spoken. Not learner-facing.

## 0. VS Code setup (before you hit record)

Open the files you will show on camera (run from the repo root):

```bash
code exercises/13-cost-monitoring/cost-monitoring-starter/DEMO.md exercises/13-cost-monitoring/cost-monitoring-starter/INSTRUCTIONS.md exercises/13-cost-monitoring/cost-monitoring-starter/INTERFACES.md exercises/13-cost-monitoring/cost-monitoring-starter/src/pricing.py exercises/13-cost-monitoring/cost-monitoring-starter/src/generator.py exercises/13-cost-monitoring/cost-monitoring-starter/src/cost/tracker.py exercises/13-cost-monitoring/cost-monitoring-starter/src/cost/dashboard.py exercises/13-cost-monitoring/cost-monitoring-starter/src/models.py exercises/13-cost-monitoring/cost-monitoring-starter/scripts/seed_cost_log.py
```

Files and why each is on screen:
- `DEMO.md`: the walkthrough you reference; wires `compute_cost`, logs a row, renders the dashboard.
- `INSTRUCTIONS.md`: the three exercises; keep it open to point at each acceptance criterion.
- `INTERFACES.md`: the frozen signatures, so "do not change the signature" is concrete.
- `src/pricing.py`: the per-model rate table and `compute_cost`; the asymmetry lives in this dict.
- `src/generator.py`: where token counts come off the response and become a real dollar cost.
- `src/cost/tracker.py`: `log_request`, `load_log`, and `summarize`; the JSONL log and rollup.
- `src/cost/dashboard.py`: the renderer and the `/cost-dashboard` route a gateway can mount.
- `src/models.py`: the `TokenUsage` model the cache-aware bonus extends.
- `scripts/seed_cost_log.py`: the seeder behind `make seed-cost-log` for Exercise 1.
- Note: `data/cost_log.jsonl` does not exist yet; you create it when you run the seed.

## 1. Intro  (~45-60s)

*[Stage: DEMO.md on screen.]*

In this module you make the cost of your retrieval-augmented generation app, your RAG app, visible. Right now every query spends real money you can't see. By the end you'll have a log that records the cost of every call, a dashboard that rolls it up, and two alerts that protect you before the bill surprises you.

This is the difference between guessing what you spend and knowing it. You won't write much code. The real skill is reading the numbers and deciding when to act.

## 2. Topic overview  (~60-90s)

*[Stage: open src/pricing.py and point to the MODEL_PRICING table.]*

Cost monitoring has four layers. A per-request log records each call. An aggregation rolls those calls up by model and by day. An alert watches for spikes. And a pre-call estimate gates spending before a request even leaves your machine.

It starts with one small function. `compute_cost` takes the model name and the token counts, looks up the rate, and returns a dollar figure.

Here's the one idea to hold onto, and it's the misconception that trips everyone up. Cost is wildly uneven across models. A complex-tier model can cost roughly twenty to forty times what a mini-tier one costs per request. So a handful of expensive calls can dominate your entire bill. The request count lies to you; the dollars are what matter.

## 3. Exercise call-outs

### Exercise 1: Seed fifty entries and find the costliest query type

*[Stage: switch to INSTRUCTIONS.md, Exercise 1; point to scripts/seed_cost_log.py.]*

The first exercise seeds fifty synthetic rows so the rollups have something to chew on, without paying for fifty real queries. You run `make seed-cost-log`, then aggregate the log by model, by day, and by query type.

Here's what to watch out for. The complex tier is only twenty percent of the requests but close to ninety percent of the dollars. That's the asymmetry made concrete. So rank by cost, not by count, or you'll chase the wrong thing.

You're done when you've pasted three breakdowns into your writeup and named the costliest query type, with every number traceable to a row in the log.

### Exercise 2: Instrument twenty real queries and watch the log grow

*[Stage: point to log_request in src/cost/tracker.py.]*

The second exercise fires twenty real queries through the pipeline and confirms the log captures each one. You call `run_pipeline`, then `log_request`, and one row lands per call. This proves the instrumentation works end to end.

Here's the watch-out. The `query_type` you pass is just a label on the log row. It does not pick the model. The router decides that on its own. So if you tag something "simple" but the router sends it to the complex tier, the cost will surprise you. Read the model field, not your label, to know what you actually paid for.

You're done when the log grew by exactly twenty rows and your new summary is captured.

### Exercise 3: Build an alert and a pre-call budget gate

*[Stage: scroll to Exercise 3; keep INSTRUCTIONS.md in view.]*

The third exercise builds two protections. First, a rolling-baseline alert that warns when today's cost is more than twice the recent average. Notice it uses a multiple, not a fixed dollar amount, because the multiple catches a regression at any traffic level.

Then a pre-call gate. It uses tiktoken to estimate a request's cost before sending it, and refuses anything over your limit.

Here's the watch-out. That estimate lands within roughly ten to thirty percent of the real cost, never exact, because you can't know the answer's length until the model writes it. That's fine. The gate is about the mechanism, refusing pathological calls, not billing to the cent.

You're done with a working alert, a working gate, and one reconciliation showing the percentage gap.

## 4. Key insights  (~30-45s)

Three takeaways. First, count the dollars, not the requests, because a few complex-tier calls can own most of your bill. Second, the cost log is ground truth for what you spent; the pre-call estimate is only a gate, so never use it for billing. Third, the label you log is a label, not a lever. The router picks the model, so always check what actually ran.

## 5. Outro  (~20-30s)

*[Stage: return to DEMO.md.]*

That's cost monitoring: a log, a rollup, an alert, and a gate. You can see what you spend and stop a runaway before it happens. The next module cuts that cost a different way, by caching answers so repeat questions never reach the model at all. See you there.
