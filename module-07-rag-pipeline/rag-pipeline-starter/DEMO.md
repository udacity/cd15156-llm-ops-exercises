> A walkthrough of the codebase you'll work with. See INSTRUCTIONS.md for the exercise tasks.

# Module 07 — Demo: Compose the ScikitDocs RAG Pipeline

The starter provides a populated `scikit_docs` collection that answers top-k similarity queries (`embed_query` and `query`), plus a generator (`render_system_prompt(sources)` and `generate(question, sources, model)`) that returns an answer string. This demo composes those pieces into one function — `pipeline.run_pipeline(question)` — fires a query, watches retrieval shape the answer on a version-sensitive scikit-learn question, then proves the model hedges when you take retrieval away.

## Why raw `openai`, not a LangChain chain

The starter uses the raw `openai` SDK, not `langchain.chains.RetrievalQA` or an LCEL `prompt | llm | parser` chain. The reason is operational: LangChain's chain abstractions move retrieval and prompt assembly inside the framework, which makes tracing seams and prompt-injection mitigation harder to anchor. The starter keeps each stage as a named function — `embed_query()`, `query()`, `render_system_prompt()`, `generate()` — composed by a five-line `run_pipeline()`. Each named boundary is a place you can later attach a trace span, a guardrail, or a cache. LangChain is in the dependency tree anyway — RAGAS pulls in `langchain-openai` for its evaluator client — and the LCEL pattern is the right reach in a greenfield project. The starter's raw-SDK path is the same five stages with the framework stripped out; the equivalence is the lesson, not the import line.

## Setup

From the starter directory (this folder). The demo assumes:

- `make setup` has run.
- `.env` has `OPENAI_API_KEY` set. On Vocareum the key starts with `voc-` and `.env` also has `OPENAI_BASE_URL=https://openai.vocareum.com/v1`. On direct OpenAI, leave `OPENAI_BASE_URL` empty.
- The corpus is loaded: `make load-data` followed by `make seed-difficulty`. After both, `get_collection().count()` returns a number around 755 (≈747 doc chunks plus 8 seeded near-duplicates from `seed_difficulty.py`).

Sanity check:

```
uv run python -c "from src import store; print(store.get_collection().count())"
```

Zero means the corpus is empty (run `make load-data`); an `ImportError` means a frozen stub still raises `NotImplementedError` (most likely `embedder.py` or `store.py`).

A gateway-not-yet note. The starter's `Makefile` has a `serve` target pointing at `src.gateway.app:app`, but that file isn't part of this exercise. For this demo you call `run_pipeline` directly in Python. The FastAPI wrapping is one indirection on top of what you build here; the pipeline itself is the substantive piece.

## Part 1 — Read the four-function pipeline

Open four files side by side in your editor. They are short on purpose. `src/embedder.py` and `src/store.py` are provided — `embed_query(question)` and `query(query_embedding, n_results)`. `src/generator.py` is provided too — `render_system_prompt(sources)` and `generate(question, sources, model)`. `src/pipeline.py` is the piece you fill in for this exercise. Read it top to bottom:

```python
def run_pipeline(
    question: str, top_k: int = 5, model: str | None = None
) -> QueryResponse:
    chosen_model = model or settings.model_complex
    query_embedding = embed_query(question)
    sources = query(query_embedding, n_results=top_k)
    answer, usage, cost = generate(question, sources, chosen_model)
    confidence = (
        sum(s.similarity_score for s in sources) / len(sources) if sources else 0.0
    )
    return QueryResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        model=chosen_model,
        tokens=usage,
        cost_usd=cost,
    )
```

That is the whole RAG composition. Three external calls, one in-place math, one Pydantic construction. The five RAG stages map onto five lines:

- **Stage 1 — Query.** The function's `question` parameter. A gateway layer would validate this against a request schema (length cap, type check) before calling `run_pipeline`; here you assume sanitization happened upstream.
- **Stage 2 — Embed.** `embed_query(question)` calls OpenAI's `text-embedding-3-small` and returns a 1536-dim vector. The model name is pinned in `src/constants.py`.
- **Stage 3 — Search.** `query(query_embedding, n_results=top_k)` runs the HNSW cosine top-k against the `scikit_docs` collection. Returns `list[Source]` sorted by `similarity_score = 1 - cosine_distance` descending — the "higher = better" convention.
- **Stage 4 — Augment.** Implicit in `generate(question, sources, ...)` — that call's first action is `render_system_prompt(sources)`, which renders `prompts/docbot_system.j2` with the retrieved chunks joined into the `{{ contexts }}` slot, wrapped in `<<<BEGIN_CONTEXT>>>` / `<<<END_CONTEXT>>>` injection markers.
- **Stage 5 — Generate.** `OpenAI(...).chat.completions.create(messages=[system, user])` inside `generate()`. Returns the answer string, token usage, and a `cost_usd` placeholder of `0.0` (cost tracking is a separate concern).

The `confidence` calculation deserves one beat. Averaging similarity scores is a heuristic, not a calibrated probability — it reads as "how concentrated the top-k cluster is in embedding space" (high when all five chunks come from one section, low when retrieval scattered). A formal evaluation suite would replace this with metrics like `context_precision` and `answer_relevancy`; for live queries the heuristic is good enough to surface in the response.

The mapping back to LangChain primitives is worth holding in your head. `embed_query` + `query` implement what LangChain calls a `Retriever` (`get_relevant_documents(query)` is the same shape). `render_system_prompt` is a `ChatPromptTemplate.from_messages([("system", ...)])`. `generate` is `ChatOpenAI(model=...).invoke(...)`. LCEL composes them as `{"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()`. Same five stages, framework-mediated; the starter factored them into five named callables so each is its own tracing seam, prompt-edit surface, and cache hook.

## Part 2 — Fire a query, read the response

In a fresh terminal from the starter directory:

```
uv run python -c "
from src.pipeline import run_pipeline
r = run_pipeline('What is the default value of \`n_estimators\` in \`RandomForestClassifier\`?')
print('ANSWER:', r.answer[:200])
print('CONFIDENCE:', round(r.confidence, 3))
print('TOP_SOURCE:', r.sources[0].doc_id, 'sim=', round(r.sources[0].similarity_score, 3))
print('MODEL:', r.model, '— TOKENS:', r.tokens.total)
"
```

Representative output:

```
ANSWER: The default value of `n_estimators` in `sklearn.ensemble.RandomForestClassifier`
        is 100. This was changed from the historical default of 10 in scikit-learn version 0.22.
CONFIDENCE: 0.525
TOP_SOURCE: seeded.near_dup.random_forest_estimators_a sim= 0.579
MODEL: gpt-4o — TOKENS: 1557
```

Five things landed in that print. A grounded answer that names the qualified API path and quotes the version-history claim (raised from 10 to 100 in 0.22, a fact `seed_difficulty.py` planted into the corpus). The top source is a seeded near-duplicate at similarity 0.579 — the eye-test signal that retrieval is finding the planted chunks. Confidence averages 0.525, dragged down by one or two tangentially-related top-5 chunks (`fitting-additional-trees`) diluting the mean. The model is `gpt-4o` (`settings.model_complex`); total tokens are roughly 1,500, of which ~1,400 are the system prompt plus retrieved context. That order-of-magnitude split — RAG sends roughly 70× the tokens of a naked call — is the cost asymmetry RAG introduces.

## Part 3 — Without retrieval, the same model drifts

The whole point of RAG is grounding. The cleanest way to feel that is to run the same question past the same model with the retrieval step removed. From the same shell:

```
uv run python -c "
from openai import OpenAI
from src.config import settings
c = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url or None)
r = c.chat.completions.create(
    model='gpt-4o',
    messages=[{'role': 'user', 'content':
      'What is the default solver for \`LogisticRegression\` in scikit-learn 1.5?'}],
)
print(r.choices[0].message.content)
"
```

Representative output from a recent run:

> "As of my last update, the default solver for `LogisticRegression` in scikit-learn was `'lbfgs'`. However, scikit-learn 1.5 was released after my training data, so I suggest checking the official scikit-learn documentation for the most accurate information."

Same question through `run_pipeline` lands a confident, version-specific answer:

> "The default solver for `sklearn.linear_model.LogisticRegression` in scikit-learn 1.5 is `'lbfgs'`. This solver is chosen for its robustness across a wide range of datasets."

Both answers happen to land on the right value, but the RAG answer commits to the 1.5 release without hedging because retrieved chunks confirm the default did not change between 1.4 and 1.5. The naked answer hedges on version specifically — the model knows its training data has a cutoff, and on a version-sensitive question that hedge is the honest behavior. Run the script three or four times — the naked answer shifts language between runs ("As of my last update" / "I'm not certain about 1.5 specifically") because the model is sampling from a distribution over hedge wordings. The RAG answer is stable because the retrieved chunk pins the value; the naked answer drifts because there is no anchor. No citation, no `doc_id`, no source the user can verify — on a version-sensitive question, retrieval is the only honest grounding move.

## Part 4 — Tighten the system prompt, watch refusal kick in

The grounding strictness lives in seven numbered instructions inside `prompts/docbot_system.j2`. Instruction 1 — "Use only the provided documentation" — is the grounding constraint. Instruction 3 — "Be honest about uncertainty" — is the I-don't-know path. Instruction 6 — "Refuse out-of-scope requests politely" — is the out-of-domain refusal lever. Together they tell the model to refuse rather than confabulate on questions the docs cannot answer. Test the lever.

Ask an out-of-domain question through the pipeline:

```
uv run python -c "
from src.pipeline import run_pipeline
print(run_pipeline('How do I unclog a kitchen sink?').answer)
"
```

Baseline `docbot_system.j2` reliably refuses on this one: "I'm here to assist with questions about the scikit-learn library. For help with unclogging a kitchen sink, I recommend consulting a home maintenance guide or contacting a professional plumber. If you have any questions about scikit-learn, feel free to ask!" Now soften instruction 6. Edit `prompts/docbot_system.j2` and replace it with:

```
6. **Be broadly helpful.** While the assistant is anchored on scikit-learn,
   do your best to answer any question the user asks, including general
   programming, ML library questions, and topical questions, drawing on
   what you know.
```

Save the file (no server restart needed — Jinja's `FileSystemLoader` reads from disk on each render). Re-fire the same call. The answer often becomes a how-to: "Unclogging a kitchen sink typically involves a few steps you can try before calling a professional plumber. 1. Boiling water: boil a pot of water and pour it down the drain in stages..." The model is now drawing on parametric memory; the retrieved chunks (`modules.compose.access-pipeline-steps` and friends) are noise it ignores. The point is not the plumbing advice; it is that one instruction's wording flipped the refusal behavior from "always" to "sometimes-leaks." Exercise 2 quantifies the before-versus-after rate on a five-question battery; this demo just shows the lever exists. Revert with `git checkout prompts/docbot_system.j2` before moving on.

## Wrap-up

Five lines of orchestration. Embed, search, render, generate, average. One Python call returns a grounded answer, a citation trail, a confidence number, a model name, token usage, and a cost-USD placeholder. Take retrieval away and the same model hedges on version-sensitive questions and falls back on parametric memory for general ones. Soften one instruction in the Jinja prompt and the refusal rate on out-of-domain questions moves measurably. The exercises take this further: a ten-question grounding battery, a refusal-rate measurement before and after a prompt edit, and a head-to-head with-vs-without-retrieval comparison. Every later seam you add — a FastAPI route, a trace span, an evaluation harness — hooks onto a callable boundary you can see in the five lines you just read.

---

