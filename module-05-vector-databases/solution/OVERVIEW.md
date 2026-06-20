---
module_number: 5
module_title: "Configure a Vector Database with Chroma and sentence-transformers"
slug: vector-databases
unit_type: overview-video-script
target_minutes: "5-7"
word_count: 701
---

# Module 05 Overview Video: Vector Databases with Chroma

> Recording aid for a 5-7 min overview (read pace ~100 wpm). Read all prose aloud, top to bottom. `*[Stage: ...]*` lines are screen directions, not spoken. Not learner-facing.

## 0. VS Code setup (before you hit record)

Open the files you will show on camera:

- `../vector-databases-starter/DEMO.md`
- `../vector-databases-starter/INSTRUCTIONS.md`
- `../vector-databases-starter/INTERFACES.md`
- `../vector-databases-starter/src/chunker.py`
- `../vector-databases-starter/src/embedder.py`
- `../vector-databases-starter/src/store.py`
- `../vector-databases-starter/src/models.py`

Files and why each is on screen:
- `DEMO.md`: the walkthrough you will reference; shows the three-stub build end to end.
- `INSTRUCTIONS.md`: the three exercises; keep it open to point at each acceptance criterion.
- `INTERFACES.md`: the frozen signatures for `chunk_doc`, `embed`/`embed_query`, and the store functions. This is what "do not change the signature" means.
- `src/chunker.py`: the section-header chunker; the two-tier size decision and the `chunk_id` postfix live here.
- `src/embedder.py`: the batched embedder; the model name you will swap in Exercise 3 is here.
- `src/store.py`: the Chroma wrapper; the one cosine line is the most important line in the module.
- `src/models.py`: the `Source` model (`doc_id`, `chunk_text`, `similarity_score`) so the query return type is concrete.
- Optional, if asked: `src/corpus.py` (the document parser feeding the chunker), `src/config.py` and `src/constants.py` (model name, dimensions, Vocareum bridge).

## 1. Intro  (~45-60s)

*[Stage: DEMO.md on screen.]*

In this module you build the part of a RAG app that actually finds things: the vector store. RAG is short for retrieval-augmented generation. By the end you'll have built a vector collection from scratch, measured how good its retrieval is, and swapped the embedding model to weigh the trade-off.

This is the retrieval layer everything else depends on. If it returns the wrong text, no prompt and no model can save the answer. The good news: you won't write much code. The real skill is judging whether the results are good.

## 2. Topic overview  (~60-90s)

*[Stage: bring the three source files forward: src/chunker.py, src/embedder.py, src/store.py.]*

A vector database stores meaning, not words. Each chunk of text becomes a vector, and search means finding the vectors that point closest to your question. That's why a question about scaling features can match a section that never says "scale."

Three small pieces make the layer: a chunker splits documents at section boundaries, an embedder turns text into vectors, and a store indexes them for nearest-neighbor search.

Here's the one idea to hold onto. The similarity score is the cosine of two vectors: one point zero means identical, zero means unrelated. It's a geometry measurement, not a confidence percentage. And here's the misconception that trips up everyone: a high score doesn't mean the answer is correct. It only means the text is close in wording. The rest of this course lives in the gap between those two ideas.

## 3. Exercise call-outs

### Exercise 1: Build and verify the collection

*[Stage: switch to INSTRUCTIONS.md, Exercise 1.]*

The first exercise builds the collection from an empty directory. Picture the ingestion as five stages: clone, parse, embed, write to the store, and record a version file. Only the embed stage costs money; a cache makes later rebuilds free.

Here's what to watch out for. If a build you expected to be cold finishes in under a second, you didn't really rebuild; you're looking at a stale collection. Delete the `data/chroma` directory and run the load again. To complete this exercise, rebuild the collection from scratch and confirm the document count the load command prints matches what Chroma actually stored.

### Exercise 2: Measure recall@5 on a 12-question golden subset

*[Stage: scroll to Exercise 2; keep the reference results table in view.]*

The second exercise measures retrieval quality with recall at five: did the right section appear in the top five.

Here's the heart of the module, and the watch-out. A perfect twelve out of twelve isn't a victory. On a corpus this small, top-five scoring saturates and can't tell two embedders apart. The signal that matters is hit-at-one: is the very first result right? You'll even see deliberately confusing seeded chunks rank first, because a dense model rewards text that mirrors the question. So recall stays high while the answer quietly drifts. That gap is why a later module scores faithfulness on its own.

One practical note: run the script with the `PYTHONPATH=.` prefix, or the imports fail. To complete this exercise, get the recall script running with the correct PYTHONPATH and have it report a recall at five of at least zero point seven.

### Exercise 3: Swap to a local sentence-transformers embedder

*[Stage: open src/embedder.py and point to the model name; this is the one line you change.]*

The third exercise swaps the hosted embedder for a local model called MiniLM, and asks if the trade is worth it. First, the key concept: a store locks its vector dimension at the first insert. The hosted model makes a 1536-dimensional vector; the local one makes 384. You can't query one with the other.

So here's what to watch out for. Changing the model is one line, but the rebuild isn't optional. Query the old collection with the new model and you get an `InvalidDimensionException`. That error is the lesson, not a bug. The fix is to rebuild into a separate collection, then swap which one the app reads. That's the blue-green pattern a later module builds on. To complete this exercise, produce a side-by-side table where both tie on recall at five but separate on hit-at-one.

## 4. Key insights  (~30-45s)

*[Stage: point to the cosine line in src/store.py (the `hnsw:space` metadata).]*

Three takeaways. First, retrieval quality isn't one number: recall can read perfect while the top result is wrong, so always look one rank deeper. Second, the cosine setting and unit-normalized vectors are what make the score mean anything, so that one line of configuration is load-bearing. Third, swapping an embedder is always a rebuild and a swap, never an edit in place, because the dimension is locked at the first insert.

## 5. Outro  (~20-30s)

*[Stage: return to DEMO.md.]*

That's the retrieval layer: a chunker, an embedder, and a store pinned to cosine. The next module wires it into the RAG pipeline that turns retrieved chunks into grounded answers. A later module checks whether they stay faithful. See you in the next one.
