---
module_number: 9
module_title: "Implement LLM Call Tracing with Phoenix"
slug: observability-tracing
unit_type: overview-video-script
target_minutes: "5-7"
word_count: 689
---

# Module 09 Overview Video: Observability and Tracing with Phoenix

> Recording aid for a 5-7 min overview (read pace ~100 wpm). Read all prose aloud, top to bottom. `*[Stage: ...]*` lines are screen directions, not spoken. Not learner-facing.

## 0. VS Code setup (before you hit record)

Open the files you will show on camera:

- `../observability-tracing-starter/DEMO.md`
- `../observability-tracing-starter/INSTRUCTIONS.md`
- `../observability-tracing-starter/INTERFACES.md`
- `../observability-tracing-starter/src/tracing.py`
- `../observability-tracing-starter/src/pipeline.py`
- `../observability-tracing-starter/scripts/seed_traces.py`
- `../observability-tracing-starter/scripts/show_traces.py`

Files and why each is on screen:
- `DEMO.md`: the walkthrough you will reference; wires Phoenix in and reads one six-span trace end to end.
- `INSTRUCTIONS.md`: the three exercises; keep it open to point at each acceptance criterion.
- `INTERFACES.md`: the frozen signatures, so learners know which contracts the tracing layer must not break.
- `src/tracing.py`: the tracing surface; `init_tracing` and `traced_pipeline` live here, and Exercise 3 edits this file.
- `src/pipeline.py`: the plain RAG composition; show it to contrast the untraced path with the traced wrapper.
- `scripts/seed_traces.py`: the seed script behind `make seed-traces`; the five-question pack lives at the top.
- `scripts/show_traces.py`: the raw-span reader; Exercise 3 uses its `_fetch_spans` to confirm a custom attribute.

## 1. Intro  (~45-60s)

*[Stage: DEMO.md on screen.]*

In this module you make your retrieval-augmented generation app, your RAG app, observable. You wire in tracing so every request leaves a record you can read.

By the end you'll have fired a traced query, opened one trace, exported an evidence file, and added your own attribute to the trace.

Here's why this matters. When an answer is slow or wrong, a log line tells you it happened. A trace tells you where. That's the difference you're building.

## 2. Topic overview  (~60-90s)

*[Stage: bring src/tracing.py forward; point to the span tree comment in traced_pipeline.]*

A trace is the story of one request. It's a tree. Each node in that tree is a span, and a span is one timed step: retrieve, embed, search, generate.

The backend here is Arize Phoenix. It runs in-process, with no account and no separate server, and it shows you the trace at localhost colon 6006.

Now the one idea to hold onto. Every span carries its own attributes, its own little bag of facts. So you have to know which span holds which value. The top similarity score lives on the search span, not the retrieve span. The token counts live on the generate span. Reach for a value on the wrong span and you'll find nothing.

Here's the misconception to drop. Tracing isn't extra logging you sprinkle around. You name your spans once, you read attributes by name, and the whole stack downstream reads from that same tree.

## 3. Exercise call-outs

### Exercise 1: Read one trace in the Phoenix UI

*[Stage: switch to INSTRUCTIONS.md, Exercise 1.]*

The first exercise builds one habit: look at a trace before you trust any dashboard. You'll fire three queries, open one trace in Phoenix, and write up what you see.

Here's what to watch out for. Phoenix only lives while the Python process that started it is alive. Fire your queries and let the process exit, and the user interface is gone with it. Start the session with `python -i` so it stays up while you click around.

One more. Phoenix logs every OpenAI call as its own trace, so the embedding calls clutter the list. Filter on the name `rag_query` to see only the full request. To complete this exercise, print three trace IDs, plus a paragraph naming which span dominated the time.

### Exercise 2: Export the rubric section 7 evidence file

*[Stage: open scripts/seed_traces.py; point to the question pack at the top.]*

The second exercise is the one to weight your time toward. Verification here is in-process and headless. There's no browser. The trace IDs plus the exported file are your no-browser substitute for clicking around a user interface.

You run `make seed-traces`. It fires a five-question pack, waits for the exporter to flush, and writes a table to `data/trace_evidence.md`. Each row names the slowest child span for that request.

Watch the slowest-child column, because it carries the lesson. The slowest step is almost always generation. The trace is how you prove that rather than guess it. When something other than generate tops the column, retrieval got slow or the prompt grew. To complete this exercise, hand in the five-row table and two paragraphs reading specific rows.

### Exercise 3: Add a custom span attribute

*[Stage: open src/tracing.py; point to the retrieve span block where rag.sources.count is set.]*

The third exercise grows the schema. You add one attribute, the top retrieval score, to the retrieve span.

Here's the watch-out, and it's the whole module in one trap. A span closes when its `with` block ends. Set your attribute after the block and it goes nowhere. So the new line has to sit inside the retrieve block, not after it.

To check your work, fire the query and dump the spans in the same process, because Phoenix's store is per-process and vanishes when the process exits. To complete this exercise, produce a four-line diff and prove the attribute landed, with the test suite still passing.

## 4. Key insights  (~30-45s)

*[Stage: return to the span tree comment in src/tracing.py.]*

Three takeaways. First, a trace is a tree of spans, and each value lives on exactly one span, so always ask which span before you read an attribute. Second, the trace IDs and the exported file are your evidence when there's no browser to click. Third, the slowest step is almost always generation, and the trace is how you prove it instead of guessing.

## 5. Outro  (~20-30s)

*[Stage: return to DEMO.md.]*

That's observability: name your spans, read by name, and grow the schema when you need to. The evaluation, cost, and latency modules all read from this same tree, so the wiring you do here is the backbone for everything operational that follows. See you in the next one.
