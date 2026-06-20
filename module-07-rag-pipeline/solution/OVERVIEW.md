---
module_number: 7
module_title: "Compose the ScikitDocs RAG Pipeline"
slug: rag-pipeline
unit_type: overview-video-script
target_minutes: "5-7"
word_count: 683
---

# Module 07 Overview Video: The RAG Pipeline

> Recording aid for a 5-7 min overview (read pace ~100 wpm). Read all prose aloud, top to bottom. `*[Stage: ...]*` lines are screen directions, not spoken. Not learner-facing.

## 0. VS Code setup (before you hit record)

Open the files you will show on camera:

- `../rag-pipeline-starter/DEMO.md`
- `../rag-pipeline-starter/INSTRUCTIONS.md`
- `../rag-pipeline-starter/INTERFACES.md`
- `../rag-pipeline-starter/src/pipeline.py`
- `../rag-pipeline-starter/src/generator.py`
- `../rag-pipeline-starter/prompts/docbot_system.j2`
- `../rag-pipeline-starter/src/embedder.py`
- `../rag-pipeline-starter/src/store.py`
- `../rag-pipeline-starter/src/models.py`
- `../rag-pipeline-starter/src/config.py`
- `../rag-pipeline-starter/data/golden_set.csv`
- `../rag-pipeline-starter/data/negative_set.csv`

Files and why each is on screen:
- `DEMO.md`: the walkthrough you will reference; it composes the four callables into `run_pipeline` and fires a query end to end.
- `INSTRUCTIONS.md`: the three exercises; keep it open to point at each acceptance criterion.
- `INTERFACES.md`: the frozen signatures, so the learner sees what "do not change the signature" means for `run_pipeline`, `query`, and `generate`.
- `src/pipeline.py`: the five-line composition you fill in; this is the heart of the module.
- `src/generator.py`: the `generate` call and `render_system_prompt`, where retrieved chunks become the prompt.
- `prompts/docbot_system.j2`: the system prompt; instruction six is the refusal lever Exercise 2 edits.
- `src/embedder.py`: `embed_query`, the embed stage that turns a question into a vector.
- `src/store.py`: `query`, the search stage that returns the top chunks.
- `src/models.py`: the `QueryResponse` and `Source` types, so the return shape is concrete.
- `src/config.py`: the settings the naked OpenAI call reads in Exercise 3.
- `data/golden_set.csv` and `data/negative_set.csv`: the in-domain and off-topic questions the exercises draw from.

## 1. Intro  (~45-60s)

*[Stage: DEMO.md on screen.]*

In this module you assemble the whole thing: a retrieval-augmented generation pipeline, RAG for short. You've already got a vector store that finds relevant text and a generator that writes an answer. Here you wire them into one function, `run_pipeline`, that takes a question and returns a grounded answer.

It's only five lines: embed the question, search the store, hand the chunks to the model, average the scores, and return. The skill isn't the wiring. It's judging whether retrieval actually made the answer better.

## 2. Topic overview  (~60-90s)

*[Stage: open src/pipeline.py, point to the five lines of run_pipeline.]*

Here's the core idea. Retrieval-augmented generation means you fetch relevant documents first, then ask the model to answer using only those documents. The point is grounding. When the retrieved text doesn't support an answer, you want the model to refuse honestly rather than invent one. An honest "I don't know" beats a confident wrong answer every time.

*[Stage: point to the embed, search, generate lines in turn.]*

Five stages map onto five lines. Embed turns the question into a vector. Search finds the closest chunks. Augment folds those chunks into the prompt. Generate calls the model. And a confidence number averages the similarity scores.

Now the misconception to clear up. RAG reduces hallucination, but it does not eliminate it. The model can still drift, lean on memory, or get diluted by a weakly-related chunk. Retrieval stacks the odds in your favor; it doesn't guarantee the outcome.

## 3. Exercise call-outs

### Exercise 1: A ten-question grounding battery

*[Stage: switch to INSTRUCTIONS.md, Exercise 1; point at data/golden_set.csv and data/negative_set.csv.]*

The first exercise builds your eye. You fire ten questions through the pipeline, five in-domain and five off-topic, then read each answer and label it grounded, partial, or hallucinated. Reading the source chunk yourself is the whole skill, because a model asked to grade its own answer almost always says it's fine.

Here's what to watch for. The interesting answer is usually the one that refuses when you didn't expect it. When the retrieved chunk doesn't contain the literal default value, a well-behaved model declines to commit, and that honest refusal is the lesson, not a miss. One practical note: run the script with the `PYTHONPATH=.` prefix, or the imports fail. To complete this exercise, build a ten-row table and two tally counts.

### Exercise 2: Measure refusal rate before and after a prompt edit

*[Stage: open prompts/docbot_system.j2, point to instruction six.]*

The second exercise quantifies the refusal lever. You run five off-topic questions against the baseline prompt, count the clean refusals, then soften instruction six to be permissive, re-run, and count again. Two numbers, before and after.

Here's the watch-out. Some refusals survive the edit no matter what you write, because the model self-censors on questions it has no live data for, like today's weather. That's a different failure mode from the prompt, and naming it correctly is the point. One discipline that matters: revert the prompt with `git checkout` before you move on, or the edit leaks into every later module. To complete this exercise, report two refusal rates and two example answers that flipped.

### Exercise 3: With versus without retrieval, side by side

*[Stage: open src/config.py, point to the settings the naked call reads; keep INSTRUCTIONS.md Exercise 3 in view.]*

The third exercise is the case for the whole architecture. You fire five questions twice: once through `run_pipeline` with retrieval on, once through a naked model call with no context. Comparing the grounded answer against the naked one on the same question is how you actually feel the difference retrieval makes.

Here's the honest part. You'll label each pair materially similar or materially different, and that's a judgment call, so your exact numbers may differ from the reference run. On a couple of questions retrieval clearly wins, like a version-specific fact the model couldn't have memorized. On others the model's own knowledge is good enough and retrieval changes nothing. To complete this exercise, end with a tally and one sentence on when retrieval matters most.

## 4. Key insights  (~30-45s)

*[Stage: return to src/pipeline.py.]*

Three takeaways. First, the goal of RAG is grounding: the model should refuse when the context doesn't support an answer, not confabulate one. Second, retrieval lowers hallucination but never zeroes it, so you always measure against the naked model to feel the real lift. Third, one line in the system prompt controls refusal behavior, which makes it the most operationally important lever in the stack.

## 5. Outro  (~20-30s)

*[Stage: return to DEMO.md.]*

That's the pipeline: embed, search, augment, generate, average, in five readable lines. Every later layer hooks onto those same boundaries. The next modules wrap them in tracing spans, a formal evaluation suite, and a gateway route. See you in the next one.
