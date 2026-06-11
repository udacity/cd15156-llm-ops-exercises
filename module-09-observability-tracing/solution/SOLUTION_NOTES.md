# Solution notes — Module 09

The only code change is the Exercise 3 edit to `src/tracing.py`:
adding one `retrieve_span.set_attribute("rag.retrieve.top_score", ...)` call inside
the `retrieve` span. Exercises 1 and 2 produce writeups and a regenerated evidence
file rather than code edits — those reference solutions live below.

## Exercise 1 — Read one trace in the Phoenix UI

Three `trace_id` values printed by the snippet (your IDs will differ — they are
freshly generated 32-character hex strings from the W3C Trace Context spec).

Example five-attribute glossary for the `RandomForestClassifier` trace:

- `rag.latency_ms = 4833.0` — total wall-clock time for the request, in ms.
  This is the headline number any latency dashboard reads first.
- `rag.confidence = 0.529` — mean cosine similarity across the five retrieved
  chunks, in `[0, 1]`. Below `CONFIDENCE_THRESHOLD = 0.7`, so a production
  system would surface a "low confidence" hedge to the user.
- `rag.cost_usd = 0.0089` — per-request OpenAI spend computed from the prompt
  and completion token counts via the pricing table in `src/pricing.py`.
- `rag.model = "gpt-4o"` — the chat-completions model the `generate` stage
  invoked. Pinned at `settings.model_complex` unless overridden per call.
- `rag.sources` — list of `(doc_id, similarity_score)` tuples for the five
  chunks the retriever returned. Useful when an answer looks wrong — you can
  open the doc_ids in the corpus and check whether retrieval surfaced the
  right pages.

Example paragraph for the span tree:

> The `generate` span dominated total duration at ~2.6 seconds of the 4.8-second
> request, which is the expected shape on `gpt-4o`: chat-completions latency
> scales with completion-token count and the `generate` call produced a
> multi-sentence answer. Inside `embed` and `generate` the OpenAIInstrumentor
> contributed auto-spans named after the underlying SDK call (`Embeddings.create`
> and `ChatCompletion.create`) carrying input messages, output messages, and
> token counts as OpenInference-shaped attributes. `retrieve` and `augment`
> together accounted for under 200 ms, confirming that for short questions
> against a well-indexed corpus, the LLM call is the entire latency budget.

## Exercise 2 — Export the rubric §7 evidence file

Run `make seed-traces`. The script writes `data/trace_evidence.md` with a
five-row table plus the "slowest step across N traces" one-liner.

Example commentary (your row numbers and durations will differ):

> **Slowest trace (row 5, comparison query)**: latency 6,420 ms, `generate`
> child took 5,810 ms — the dominant cost was a 312-token completion produced
> by the multi-API comparison prompt. The bottleneck is the model's
> token-generation speed, not retrieval or prompt assembly. Operationally,
> the path to faster comparisons is shorter completions (a tighter system
> prompt, or `max_tokens`) before considering streaming or a smaller model.
>
> **Off-topic refusal (row 4, weather query)**: latency 1,210 ms, `generate`
> child took 980 ms with a 38-token completion. The same six-span tree fires,
> but every stage is cheap because the model refuses quickly and retrieval
> returns low-similarity chunks (`rag.confidence` near 0). Useful contrast to
> the slow trace: it shows the floor of what the pipeline costs on a trivial
> query.

## Exercise 3 — Add a custom span attribute

Diff snippet for the writeup (the surrounding context is shown only for
orientation — the added lines are the four-line block):

```diff
                 with tracer.start_as_current_span("search") as search_span:
                     ...
                     search_span.set_attribute("rag.sources.count", len(sources))
+                retrieve_span.set_attribute(
+                    "rag.retrieve.top_score",
+                    max(s.similarity_score for s in sources) if sources else 0.0,
+                )
                 retrieve_span.set_attribute("rag.sources.count", len(sources))
```

Verify the attribute lands in Phoenix by re-firing one query in a fresh
`uv run python -c "..."` invocation and checking either the UI (`localhost:6006`,
click the `retrieve` span, scroll the attributes panel) or the JSON export
(`uv run python scripts/show_traces.py --json | head -50`, look for
`rag.retrieve.top_score` on the `retrieve` span entries).

The full test suite (`uv run pytest tests/`) should still pass — the new
attribute is additive and the existing tests do not pin the retrieve-span
attribute set.
