> A walkthrough of the codebase you'll work with. See INSTRUCTIONS.md for the exercise tasks.

# Module 03 — Demo: Git + Jinja2 Prompt Versioning in the ScikitDocs Starter

The concept module argued that a prompt is a versioned artifact. This demo shows the artifact in the ScikitDocs starter repo, builds the loader from a stub so you see every line that puts the prompt onto the wire, switches between two versions using Git, and runs a tiny A/B comparison so you can feel the workflow before the exercises ask you to build on it.

## Setup

You should already have the ScikitDocs starter cloned under this starter's root directory and `make setup` complete. If not, run that first; the rest of the demo will not work end-to-end without it. The demo assumes:

- this starter's root directory is your working directory for every command below.
- `.env` has `OPENAI_API_KEY` set. If you are on Vocareum, the key starts with `voc-` and `.env` also has `OPENAI_BASE_URL=https://openai.vocareum.com/v1`. If you are on direct OpenAI, leave `OPENAI_BASE_URL` empty. The starter reads both through `src/config.py` and forwards them to the OpenAI client; the same code path works in either environment, which is the whole point of routing the value through pydantic-settings.
- The vector store is loaded: `make load-data`. If you skipped it, the retrieval step will return nothing useful and the prompt's `{{ contexts }}` block will render empty.

Sanity check from this starter's root directory:

```
uv run python -c "from src.config import settings; print(repr(settings.openai_base_url))"
```

You want `''` on direct OpenAI or `'https://openai.vocareum.com/v1'` on Vocareum. Anything else — a stray trailing slash, a missing `https`, the wrong host — means your `.env` is wrong and the rest of the demo will fail in a way that is confusing to debug. Fix it now.

Start the server in a second terminal:

```
make serve
```

Leave it running. Every `curl` in the demo hits it on `localhost:8080`.

## Part 1 — Build `generator.py` from the stub

Open `src/generator.py`. In the starter it is a stub — two functions, both raising `NotImplementedError`, with their signatures frozen by `INTERFACES.md`. Filled in, the entire prompt-templating system for the ScikitDocs RAG sits in this one file. Walk through it from top to bottom; every line is doing something specific.

The first piece is the template directory and a Jinja `Environment`:

```python
_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
_env = Environment(
    loader=FileSystemLoader(_PROMPTS_DIR),
    keep_trailing_newline=True,
    autoescape=False,
)
```

Two design choices worth naming. `autoescape=False` because these are plaintext prompts going to the LLM, not HTML — escaping `&` or `{` would corrupt the message. `keep_trailing_newline=True` because Jinja strips the final newline by default, and removing it can shift tokenization on some models. Both are non-defaults; both are documented in the Jinja API reference.

The template itself lives at `prompts/docbot_system.j2`. Open it and read the first three paragraphs. The structure is a system message with one Jinja variable inside an explicit data-boundary block:

```jinja
<<<BEGIN_CONTEXT>>>
{{ contexts }}
<<<END_CONTEXT>>>
```

Everything outside the `{{ contexts }}` placeholder is fixed text. The placeholder is where retrieved scikit-learn doc chunks get spliced in at request time. The `<<<BEGIN_CONTEXT>>>` / `<<<END_CONTEXT>>>` markers are an indirect-prompt-injection mitigation — note that data and instructions are visually separated in the rendered prompt and that the template instructions explicitly tell the model to treat anything inside the markers as data, never as instructions to follow.

The render itself is one function:

```python
def render_system_prompt(sources: list[Source]) -> str:
    template = _env.get_template("docbot_system.j2")
    contexts = "\n\n---\n\n".join(s.chunk_text for s in sources)
    return template.render(contexts=contexts)
```

`Source` is the pydantic model from `src/models.py` — `doc_id`, `chunk_text`, `similarity_score`. We join chunks with a horizontal-rule separator so the model can see where one excerpt ends and the next begins; three hyphens render unambiguously where a blank line might collapse.

While you have the template open, scroll through the seven numbered instructions. Notice that the prompt encodes more than tone. Instruction 1 says "Use only the provided documentation" — that is the grounding constraint that makes RAG work. Instruction 2 mandates qualified-name citations. Instruction 3 demands explicit uncertainty. Each is a behavior decision, and each is a candidate for an A/B test the moment a stakeholder asks "what if the bot was less cautious about saying it doesn't know?" Versioning the file means you can answer that question with two commits and a diff.

Now the second function, `generate`:

```python
def generate(question: str, sources: list[Source], model: str) -> tuple[str, TokenUsage, float]:
    client = OpenAI(base_url=settings.openai_base_url or None)
    system_prompt = render_system_prompt(sources)
    response = client.chat.completions.create(
        model=model,
        temperature=constants.GENERATION_TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    answer = response.choices[0].message.content or ""
    usage = TokenUsage(
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
    )
    cost_usd = 0.0
    return answer, usage, cost_usd
```

Three lines deserve attention. `OpenAI(base_url=settings.openai_base_url or None)` is the Vocareum / direct-OpenAI bridge: when `OPENAI_BASE_URL` is the empty string the SDK falls back to its built-in default; when it is the Vocareum URL the SDK routes through the proxy. Same code, two deploy targets, zero conditional branches.

`temperature=constants.GENERATION_TEMPERATURE` imports from `src/constants.py`. The locked value is `0.2` — low enough for factual recall, not zero so the model can paraphrase. Hardcoding `0.2` in a call site is a review-blocker caught by `make consistency-check`.

`cost_usd = 0.0` is a placeholder. Cost-aware downstreams see zero until a pricing table replaces the literal with `pricing.compute_cost(usage, model)` — the slot-and-fill is intentional so the cost concern stays a separate piece without rework here.

Sanity check before you go further:

```
uv run python -c "from src.generator import render_system_prompt; from src.models import Source; print(render_system_prompt([Source(doc_id='x', chunk_text='LogisticRegression default penalty is l2.', similarity_score=0.9)])[:200])"
```

You should see the first 200 characters of the rendered system prompt with the chunk visible inside the context block. If the import fails, run `uv sync`; if the render fails, `parents[1]` is landing in the wrong place — print `_PROMPTS_DIR` to debug.

## Part 2 — Switching versions with a Git branch

The templating system you just walked is enough to render a prompt. The versioning system is what makes it auditable, which is the entire point of this module. Create a branch for a tweaked production prompt:

```
git checkout -b prompt-prod-tighter
```

Open `prompts/docbot_system.j2` and change instruction 4 from "Be concise and direct" to:

```
4. **Be terse.** Answer in two sentences maximum. Cite the qualified estimator name only when directly relevant.
```

Save, commit:

```
git add prompts/docbot_system.j2
git commit -m "prompt: terse variant for A/B comparison"
```

You now have two prompt versions, one per branch. `main` is the baseline; `prompt-prod-tighter` is the candidate. With the server running, query under the candidate branch:

```
curl -s -X POST http://localhost:8080/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the default penalty for sklearn.linear_model.LogisticRegression?"}' | uv run python -c "import sys, json; print(json.load(sys.stdin)['answer'])"
```

Expected response under `prompt-prod-tighter` is a one-to-two-sentence answer that names `l2` and stops. Switch back and re-query:

```
git checkout main
curl -s -X POST http://localhost:8080/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the default penalty for sklearn.linear_model.LogisticRegression?"}' | uv run python -c "import sys, json; print(json.load(sys.stdin)['answer'])"
```

The `main` answer is typically longer — a sentence with the default, plus a note about the `C` regularization strength or the `solver` interaction. Same question, same retrieval, two prompt versions, two answers. The version control system is doing the work; nothing changed in the application code.

In production you would tag immutable commits (`git tag prompt-prod-v2`) and move a `prod` reference rather than switching branches — the pattern the LangSmith registry docs describe and the one the concept module recommends. Branches are easier for the demo.

One thing to watch: `make serve` runs with `--reload --reload-dir src`. Templates live under `prompts/`, so the reload trigger does not fire on a `.j2` edit. The next request still picks up the new template because `_env.get_template()` consults file mtime, which changes when Git swaps the file. If a response does not change, restart `make serve`.

## Part 3 — A tiny A/B harness

Two variants, five questions each, count something measurable. This is the minimum viable A/B — small enough to feel, not large enough to ship a decision on.

Pick five questions, loop them through both branches, save the responses, count answer length. From this starter's root directory:

```
git checkout main
for q in "What is the default penalty in sklearn.linear_model.LogisticRegression?" \
         "How does sklearn.cluster.KMeans choose initial centroids?" \
         "What does sklearn.preprocessing.StandardScaler.fit_transform return?" \
         "What is the default scoring for sklearn.model_selection.cross_val_score on a classifier?" \
         "How do you set class weights in sklearn.svm.SVC?"; do
  curl -s -X POST http://localhost:8080/query \
    -H 'Content-Type: application/json' \
    -d "{\"question\": \"$q\"}" | uv run python -c "import sys, json; print(len(json.load(sys.stdin)['answer'].split()))"
done > /tmp/words-main.txt
```

Repeat under the candidate branch:

```
git checkout prompt-prod-tighter
# same loop, redirect to /tmp/words-tighter.txt
```

Compare:

```
echo "main:    $(awk '{s+=$1} END {print s/NR}' /tmp/words-main.txt) avg words"
echo "tighter: $(awk '{s+=$1} END {print s/NR}' /tmp/words-tighter.txt) avg words"
```

Representative output:

```
main:    52.4 avg words
tighter: 19.8 avg words
```

The tighter prompt cuts response length roughly sixty percent on this five-question sample. That is a real effect on the metric, and it is a real cost reduction at output-token prices — completion tokens are typically two to four times more expensive than prompt tokens on `gpt-4o`-class models, so a sixty-percent drop in output length translates roughly to a sixty-percent drop in completion cost per request.

What you cannot conclude from five questions is whether the tighter prompt also changed answer quality. The candidate might be cutting padding, or it might be cutting the qualified-name citations that make the answer auditable. To distinguish those cases you would run RAGAS faithfulness + context precision on both variants and need many more samples — a five-percent shift in a binary quality metric at ninety-five percent confidence needs on the order of ten thousand requests per variant. Five questions is enough to feel the workflow, not enough to ship a decision.

Clean up before moving on:

```
git checkout main
git branch -D prompt-prod-tighter
```

Or keep it around for the exercises, which will revisit the same A/B pattern with a real success metric.

## Wrap-up

That is the full Git + Jinja2 prompt versioning loop, end to end. A template on disk under `prompts/`. A loader in `src/generator.py` that you just built from a stub. Git branches or tags to switch versions. A tiny A/B that surfaces a measurable difference. The pieces are small and the indirection is shallow, which is the strength of this pattern — there is nothing magic between the template file and the model call.

The exercises take this further: extending the template with new variables, building an environment-based loader that selects dev or prod at runtime via a `PROMPT_ENV` setting, and running an A/B with a real success metric (refusal rate) and a chi-squared significance test from `scipy.stats`. Each exercise pins to a concrete acceptance criterion you can self-verify before moving on.
