# Module 07 — Reference Solution Notes

The three deliverables are write-ups, not code patches. The exercise scripts
under `exercises/` reproduce the printed output the learner classifies; the
expected classifications, tallies, and one-paragraph interpretations are
documented here.

These are reference outcomes from author runs against `gpt-4o` on the seeded
~755-chunk corpus. Individual learner runs will vary on the boundary cases
because `GENERATION_TEMPERATURE=0.2`.

---

## Exercise 1 — Grounding battery (10 questions)

Run: `uv run python exercises/grounding_battery.py | tee /tmp/answers.txt`

### Representative classification table

| # | Category            | Question (short)                          | Class        | Rationale (one sentence)                                                                         |
|---|---------------------|-------------------------------------------|--------------|--------------------------------------------------------------------------------------------------|
| 1 | in-domain factual   | RF `n_estimators` default                 | Grounded     | Top source is the seeded `random_forest_estimators_a` chunk; answer commits to 100 and cites the 0.22 change. |
| 2 | in-domain factual   | `StandardScaler` behavior                 | Grounded     | Answer describes zero-mean / unit-variance and names the qualified `sklearn.preprocessing` path. |
| 3 | in-domain factual   | `SVC` default kernel                      | Grounded     | Answer commits to `rbf` with top source at sim > 0.55.                                           |
| 4 | in-domain factual   | `KMeans` default `n_clusters`             | Partial      | Retrieved chunk discusses k-means but does not literally state `8`; model often hedges or refuses. |
| 5 | in-domain factual   | LogReg default solver in 1.5              | Grounded     | RAG commits to `lbfgs` with version anchor; this is the Demo Part 3 grounding example.            |
| 6 | off-topic benign    | Weather in Paris                          | Grounded*    | Clean refusal — instruction 6 directs the model to redirect on off-topic.                         |
| 7 | off-topic benign    | Unclog a kitchen sink                     | Grounded*    | Clean refusal with redirect to "consult a plumber."                                               |
| 8 | off-topic benign    | World Cup 2022 winner                     | Grounded*    | Clean refusal — model also self-censors on real-time facts.                                       |
| 9 | off-topic adjacent  | Train a transformer in PyTorch            | Grounded*    | Brief redirect; the redirect-don't-speculate clause holds on baseline.                            |
|10 | off-topic adjacent  | numpy broadcasting                        | Grounded*    | Brief redirect to numpy docs.                                                                     |

`*` Grounded for the off-topic rows means "clean refusal," per the exercise scheme.

### Tallies

- `in_domain_grounded = 4 / 5` (or 5/5 on stricter runs; the `KMeans` row is the boundary case).
- `off_topic_grounded = 5 / 5`.

### Most pedagogically interesting answer

The `KMeans` `n_clusters` question. The top retrieved chunk
(`modules.clustering.k-means.p0`) describes the k-means algorithm in general
terms but does not contain the literal default value (`8`). On most runs the
model honestly hedges or refuses rather than confabulating, which is exactly
instruction 3 ("Be honest about uncertainty") doing its job — even though
parametric memory could have produced the correct value. This is the canonical
example of a partial classification driven by retrieval gap rather than model
weakness.

---

## Exercise 2 — Refusal rate before vs after a prompt edit

Run baseline: `uv run python exercises/refusal_rate.py | tee /tmp/baseline-refusal.txt`

### Recommended edit to `prompts/docbot_system.j2`, instruction 6

Replace the existing instruction 6 with:

```
6. **Be broadly helpful.** While the assistant is anchored on scikit-learn,
   do your best to answer any question the user asks, including general
   programming, ML library questions, and topical questions, drawing on
   what you know.
```

Then re-run: `uv run python exercises/refusal_rate.py | tee /tmp/permissive-refusal.txt`

Revert when done: `git checkout prompts/docbot_system.j2`

### Representative counts

- `baseline_refusal_rate = 5 / 5` — all five off-topic questions get a clean refusal under the strict instruction 6.
- `permissive_refusal_rate = 2 / 5` — kitchen-sink, PyTorch-transformer, and numpy-broadcasting questions flip to substantive answers; weather and World Cup hold because the model self-censors on real-time facts independent of the prompt.

### Two example outputs that flipped

**Kitchen sink — baseline (refused):**
> "I'm here to assist with questions about the scikit-learn library. For help
> with unclogging a kitchen sink, I recommend consulting a home maintenance
> guide or contacting a professional plumber."

**Kitchen sink — permissive (not refused):**
> "Unclogging a kitchen sink typically involves a few steps you can try before
> calling a professional plumber. 1. Boiling water: boil a pot of water and
> pour it down the drain in stages..."

### One-sentence interpretation

Instruction 6 is load-bearing for the questions where the model has strong
parametric memory but no real-time-data dependency (kitchen sink, numpy,
PyTorch); the weather and World Cup questions are gated by the model's
training-data cutoff discipline, which the system prompt does not control.

---

## Exercise 3 — RAG vs naked, side by side

Run: `uv run python exercises/rag_vs_naked.py | tee /tmp/comparison.txt`

### Representative tally

- `materially_different = 2 / 5`
- `materially_similar = 3 / 5`

### Per-question labels and rationale

| # | Question                                      | Verdict             | Why                                                                                                |
|---|-----------------------------------------------|---------------------|----------------------------------------------------------------------------------------------------|
| 1 | LogReg default solver in 1.5                  | Materially different| RAG commits to "lbfgs in 1.5"; naked answer hedges with "as of my last update."                    |
| 2 | `SVC` default kernel                          | Materially similar  | The `rbf` default is widely documented; parametric memory matches retrieval.                       |
| 3 | What is `HistGradientBoostingClassifier`?     | Materially similar  | Public API documentation is well-represented in pretraining; both paths describe it correctly.     |
| 4 | Supervised vs unsupervised learning           | Materially similar  | Textbook ML concept; parametric memory dominates regardless of retrieval.                          |
| 5 | Population of Tokyo                           | Materially different| RAG politely refuses (off-topic); naked answers "~14 million in the 23 special wards."             |

### Wrap-up (three lines)

Retrieval matters most for the scikit-learn workload when the answer requires
a version-specific or recently-changed fact the model could not reasonably
have memorized, and on out-of-scope refusal: the system prompt only enforces
refusal when retrieval supplies grounded context, so stripping the context
collapses the refusal behavior on questions the model has any parametric
opinion about. For widely-documented stable APIs and textbook concepts,
parametric memory is competitive — retrieval is insurance against drift, not
the only path to a correct answer.
