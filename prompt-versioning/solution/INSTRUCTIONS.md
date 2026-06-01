# Module 03 — Implement a Prompt Versioning System with Git and Jinja2

## Setup (read first)

This starter is the ScikitDocs RAG application with every operational feature wired end-to-end: prompt rendering, vector retrieval, the RAG query pipeline, Phoenix tracing, RAGAS evaluation, cost monitoring, semantic caching, the FastAPI gateway, guardrails, A/B testing, RAGOps watcher, and latency optimizations. The codebase is the same one every later module builds on, so anything you do here can be exercised against a running system.

In this module you will add three things on top of that codebase: a `user_tier` Jinja conditional in `prompts/docbot_system.j2` (Exercise 1), an environment-aware prompt loader at `src/prompts/loader.py` that switches between `prompts/dev/` and `prompts/prod/` via the `PROMPT_ENV` setting (Exercise 2), and a small A/B-test harness at `scripts/ab_refusal.py` that compares refusal rates between two prompt branches with a chi-squared significance check (Exercise 3).

Run `make setup` to install dependencies, then `make load-data` once to populate the Chroma vector store. Open a second terminal and run `make serve` to start the FastAPI server on `localhost:8080` — Exercise 3's harness and the demo's curl commands both target it. Then follow the demo walkthrough first, then the three exercises in order; each exercise has a self-verifiable acceptance criterion.

---


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

Open `src/generator.py`. Before this REQ landed it was a stub — two functions, both raising `NotImplementedError`, with their signatures frozen by `INTERFACES.md`. Now it is filled in, and the entire prompt-templating system for the ScikitDocs RAG sits in this one file. Walk through it from top to bottom; every line is doing something specific.

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

Everything outside the `{{ contexts }}` placeholder is fixed text. The placeholder is where retrieved scikit-learn doc chunks get spliced in at request time. The `<<<BEGIN_CONTEXT>>>` / `<<<END_CONTEXT>>>` markers are the indirect-prompt-injection mitigation we dig into in Module 20 — for now, just note that data and instructions are visually separated in the rendered prompt and that the template instructions explicitly tell the model to treat anything inside the markers as data, never as instructions to follow.

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

`cost_usd = 0.0` is a placeholder. Module 13 (Cost Monitoring) lands `src/pricing.py` and replaces the literal with `pricing.compute_cost(usage, model)`. Until then, cost-aware downstreams see zero — the slot-and-fill is intentional so Module 13 owns the pricing table without rework here.

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
  -d '{"question": "What is the default penalty for sklearn.linear_model.LogisticRegression?"}' | jq -r .answer
```

Expected response under `prompt-prod-tighter` is a one-to-two-sentence answer that names `l2` and stops. Switch back and re-query:

```
git checkout main
curl -s -X POST http://localhost:8080/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the default penalty for sklearn.linear_model.LogisticRegression?"}' | jq -r .answer
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
    -d "{\"question\": \"$q\"}" | jq -r '.answer' | wc -w
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

What you cannot conclude from five questions is whether the tighter prompt also changed answer quality. The candidate might be cutting padding, or it might be cutting the qualified-name citations that make the answer auditable. To distinguish those cases you would run RAGAS faithfulness + context precision on both variants (Module 11) and need many more samples — a five-percent shift in a binary quality metric at ninety-five percent confidence needs on the order of ten thousand requests per variant. Five questions is enough to feel the workflow, not enough to ship a decision.

Clean up before moving on:

```
git checkout main
git branch -D prompt-prod-tighter
```

Or keep it around for the exercises, which will revisit the same A/B pattern with a real success metric.

## Wrap-up

That is the full Git + Jinja2 prompt versioning loop, end to end. A template on disk under `prompts/`. A loader in `src/generator.py` that you just built from a stub. Git branches or tags to switch versions. A tiny A/B that surfaces a measurable difference. The pieces are small and the indirection is shallow, which is the strength of this pattern — there is nothing magic between the template file and the model call.

The exercises take this further: extending the template with new variables, building an environment-based loader that selects dev or prod at runtime via a `PROMPT_ENV` setting, and running an A/B with a real success metric (refusal rate) and a chi-squared significance test from `scipy.stats`. Each exercise pins to a concrete acceptance criterion you can self-verify before moving on.

---


# Module 03 — Exercise: Build the Prompt Versioning Workflow

The demo showed the ScikitDocs starter's prompt template, built `src/generator.py` from a stub, and walked through switching versions via Git. These exercises move you from "watched it work" to "shipped it yourself." Three exercises, increasing in scope: extend the template with a new variable and a conditional, build an environment-aware loader that selects the right template at runtime, and run a real A/B test with a quality metric and a significance check. Each one has an acceptance criterion you can verify without grading help, and each one ends in a small piece of working code you will reuse in later modules. The order matters — exercise 2 builds on the template you extend in exercise 1, and exercise 3 assumes the loader pattern from exercise 2.

Plan for roughly twenty minutes total, weighted toward exercise 3 where the moving parts are real. The first two exercises are deliberately tight so you have time to actually read your A/B output and reason about it.

## Setup

You should have everything from the demo: the ScikitDocs starter cloned, `make setup` complete, `make load-data` run, `.env` configured with `OPENAI_API_KEY` (and `OPENAI_BASE_URL` if you are on Vocareum — Vocareum keys start with `voc-` and point at `https://openai.vocareum.com/v1`). If the demo's curl-to-the-running-server worked, you are set. If not, fix that before continuing; the exercises all assume a working end-to-end query path.

A second terminal running `make serve` is convenient. Exercise 3 needs the server actively responding. Exercises 1 and 2 only touch tests and can run without the server, but you may as well leave it up.

Each exercise is self-contained but they build on each other. Do them in order. If you bail out partway through one exercise, commit what you have on the branch so the next exercise starts from a known state — that is the discipline the concept module argued for, and it is the discipline these exercises practice.

## Exercise 1 — Extend the system template

The starter's `prompts/docbot_system.j2` takes one variable, `contexts`. Your job: add a second variable, `user_tier`, that conditionally injects a sentence pointing premium-tier users at the maintainer mailing list for help with questions the bot cannot answer from the docs. This is the simplest possible template change — one new variable, one Jinja conditional — and it forces you to touch the three places a template change actually lives: the template file itself, the calling code, and a test. In a real codebase you would also touch the request schema and the upstream caller, but for this exercise we keep the surface small.

The why behind the exercise: every nontrivial prompt change introduces a new variable that the application has to thread through. Forgetting to wire it through somewhere is the most common silent failure in prompt ops — the template renders, the test passes, and the variable is just always the default value because nothing ever set it. The discipline this exercise teaches is to add the variable at every layer at the same time, in one commit.

### What to do

1. Create a branch:

   ```
   git checkout -b ex1-user-tier
   ```

2. Edit `prompts/docbot_system.j2`. After the existing instruction 7 ("Format for readability"), add a Jinja conditional block:

   ```jinja
   {% if user_tier == "premium" %}
   8. **Premium-tier note.** If the question cannot be answered from the provided documentation excerpts, mention that the scikit-learn-help mailing list (https://mail.python.org/mailman/listinfo/scikit-learn) is the canonical place to ask for help on undocumented behavior.
   {% endif %}
   ```

3. Update `src/generator.py`'s `render_system_prompt` function to accept and pass `user_tier`. The function currently renders with just `contexts=contexts`; change the signature to `render_system_prompt(sources: list[Source], user_tier: str = "standard") -> str` and pass `user_tier=user_tier` through to `template.render()`. Defaulting to `"standard"` keeps existing callers working.

4. Write a unit test that confirms the template renders correctly for both tiers. Create `tests/test_prompt_tier.py`:

   ```python
   from src.generator import render_system_prompt
   from src.models import Source

   def _src(text: str) -> Source:
       return Source(doc_id="d", chunk_text=text, similarity_score=0.9)

   def test_premium_tier_includes_mailing_list():
       out = render_system_prompt(
           [_src("LogisticRegression default penalty is l2.")],
           user_tier="premium",
       )
       assert "scikit-learn-help" in out or "mailman" in out

   def test_standard_tier_omits_mailing_list():
       out = render_system_prompt(
           [_src("LogisticRegression default penalty is l2.")],
           user_tier="standard",
       )
       assert "scikit-learn-help" not in out
       assert "mailman" not in out
   ```

### Acceptance criterion

`uv run pytest tests/test_prompt_tier.py -q` passes both tests. Commit your changes on the `ex1-user-tier` branch. Bonus self-check: render the template manually in a REPL and eyeball both outputs; a passing test that produces visually broken output usually means the test was checking for the wrong substring.

### Hints

<details>
<summary>If the test fails on a <code>TypeError: render_system_prompt() got an unexpected keyword argument</code></summary>

You changed the template but not the Python function signature. Open `src/generator.py`, add `user_tier: str = "standard"` to the function parameters, and pass it through in the `.render()` call.
</details>

<details>
<summary>If both tests pass but the premium output looks malformed (extra blank lines, weird indentation)</summary>

Jinja preserves whitespace from the template literally. If you indented the `{% if %}` block, the leading whitespace shows up in the rendered output. Either un-indent the block in the `.j2` file or use Jinja's whitespace control: `{%- if user_tier == "premium" -%}` strips surrounding whitespace.
</details>

<details>
<summary>If the import in the test file fails</summary>

The test runner needs `src` on the Python path. The starter's `pyproject.toml` already configures this with `pythonpath = ["."]` under `[tool.pytest.ini_options]`. Run pytest from this starter's root directory, not from `tests/`.
</details>

## Exercise 2 — Environment-aware prompt loader

Now build the piece the concept module called out as the runtime half of versioning: a loader that picks a different template directory based on an environment variable. Production reads `prompts/prod/`; development reads `prompts/dev/`. One application, two configs.

This is the same pattern the starter uses for `OPENAI_BASE_URL` — read an environment value into a settings object, branch behavior on it without putting conditionals in every call site. The payoff is that the rest of the application code remains environment-agnostic. The cost is one new abstraction layer, which is worth it the first time you want to A/B-test a prompt in staging without redeploying production.

### What to do

1. Create a new branch from `main`:

   ```
   git checkout main
   git checkout -b ex2-env-loader
   ```

2. Create two subdirectories under `prompts/`:

   ```
   mkdir prompts/prod prompts/dev
   cp prompts/docbot_system.j2 prompts/prod/docbot_system.j2
   cp prompts/docbot_system.j2 prompts/dev/docbot_system.j2
   ```

3. Edit `prompts/dev/docbot_system.j2`. Add a verbose debugging suffix at the end:

   ```
   ## Debug mode
   This response was generated under the dev prompt. Wrap your answer with [DEV] and [/DEV] markers so we can see which template handled the request.
   ```

   Leave `prompts/prod/docbot_system.j2` unchanged.

4. Write a loader function. Create `src/prompts/loader.py`:

   ```python
   """Environment-aware prompt loader.

   Reads PROMPT_ENV from the environment (defaults to "prod") and
   returns a Jinja2 Environment rooted at prompts/<env>/.
   """
   import os
   from pathlib import Path
   from jinja2 import Environment, FileSystemLoader

   _REPO_ROOT = Path(__file__).resolve().parents[2]

   def load_environment(env_name: str | None = None) -> Environment:
       env_name = env_name or os.environ.get("PROMPT_ENV", "prod")
       if env_name not in ("dev", "prod"):
           raise ValueError(f"PROMPT_ENV must be 'dev' or 'prod', got {env_name!r}")
       prompts_dir = _REPO_ROOT / "prompts" / env_name
       if not prompts_dir.is_dir():
           raise FileNotFoundError(f"prompts directory not found: {prompts_dir}")
       return Environment(
           loader=FileSystemLoader(prompts_dir),
           keep_trailing_newline=True,
           autoescape=False,
       )
   ```

   You will need `mkdir -p src/prompts && touch src/prompts/__init__.py` first so the package is importable.

5. Write tests. Create `tests/test_prompt_loader.py`:

   ```python
   import pytest
   from src.prompts.loader import load_environment

   def test_default_is_prod(monkeypatch):
       monkeypatch.delenv("PROMPT_ENV", raising=False)
       env = load_environment()
       tmpl = env.get_template("docbot_system.j2").render(contexts="")
       assert "[DEV]" not in tmpl

   def test_dev_env_loads_dev_template(monkeypatch):
       monkeypatch.setenv("PROMPT_ENV", "dev")
       env = load_environment()
       tmpl = env.get_template("docbot_system.j2").render(contexts="")
       assert "[DEV]" in tmpl

   def test_explicit_arg_overrides_env(monkeypatch):
       monkeypatch.setenv("PROMPT_ENV", "dev")
       env = load_environment("prod")
       tmpl = env.get_template("docbot_system.j2").render(contexts="")
       assert "[DEV]" not in tmpl

   def test_invalid_env_raises():
       with pytest.raises(ValueError):
           load_environment("staging")
   ```

### Acceptance criterion

`uv run pytest tests/test_prompt_loader.py -q` passes all four tests. The function correctly switches templates based on `PROMPT_ENV`, defaults to `prod`, accepts an explicit override, and rejects bad inputs. Defaulting to `prod` rather than `dev` is intentional — accidental promotion of dev behavior to production is a category of bug worth defending against at the type level.

### Hints

<details>
<summary>If the loader can't find the prompts directory</summary>

`parents[2]` in the path resolution counts up from `src/prompts/loader.py`. That should land on this starter's root directory. Print the resolved path to verify: `print(_REPO_ROOT)`. If it lands somewhere unexpected, count the `parents` indices: `parents[0]` is the file's parent dir, `parents[1]` is `src/`, `parents[2]` is the starter root.
</details>

<details>
<summary>If you want to mirror the pydantic-settings pattern used elsewhere in the starter</summary>

`src/config.py` adds settings as class attributes on `Settings`. You could add `prompt_env: Literal["dev", "prod"] = "prod"` there, and read `settings.prompt_env` instead of calling `os.environ` directly. That keeps all configuration in one place and gives you pydantic-level validation for free. It is the pattern the starter uses for `openai_base_url`, `tracing_backend`, and `model_complex`. For this exercise either approach works; the pydantic version is cleaner if you plan to ship.
</details>

<details>
<summary>If the test <code>test_default_is_prod</code> fails on a CI environment that sets PROMPT_ENV globally</summary>

`monkeypatch.delenv(..., raising=False)` removes the variable for the test's scope, so the test is hermetic. If you wrote it without `monkeypatch.delenv` and a CI runner exports `PROMPT_ENV=dev`, the test reads that value and fails. The `monkeypatch` fixture restores the prior environment after the test, which is exactly what you want.
</details>

## Exercise 3 — A/B test with a real metric and a significance check

The demo's A/B counted average word length. That is a fine proxy for the cost dimension, but it is not a quality metric. Now you will A/B-test two prompts against a real success metric — refusal rate — and decide whether the difference is statistically significant using a chi-squared test.

The setup: variant A is the current `prompts/docbot_system.j2`. Variant B is a tweaked version that softens the "Be honest about uncertainty" instruction. The hypothesis is that the tweaked version refuses fewer questions, which may or may not be a good thing — refusing wrong is sometimes the right answer, and refusing right is too. The whole point of A/B testing is to put a number on that tradeoff rather than relying on intuition.

This exercise is the longest of the three because the real lesson is not the harness itself but reading the output honestly. You will almost certainly see a difference between variants. You will almost certainly not have enough samples to call it statistically significant. Both of those things will be true at the same time, and the discipline is to write them both down in the same report.

### What to do

1. Create the variant. Branch from `main`:

   ```
   git checkout main
   git checkout -b ex3-soft-refusal
   ```

   Edit `prompts/docbot_system.j2`, replacing instruction 3 ("Be honest about uncertainty") with:

   ```
   3. **Lean toward answering.** If you have partial information, give the best answer you can with the documentation excerpts provided. Only say you don't have information when the question is clearly outside scikit-learn.
   ```

   Commit. Do not push — local branches are fine for the exercise.

2. Write a harness. Create `scripts/ab_refusal.py`:

   ```python
   """A/B-test refusal rate between two prompt variants.

   Runs the same N questions against both branches by:
       - `git checkout main`              for variant A
       - `git checkout ex3-soft-refusal`  for variant B
   between iterations. Writes the per-question label (refused vs answered)
   to a JSON report and prints a chi-squared p-value.
   """
   import json
   import re
   import subprocess
   import sys
   import requests
   from scipy.stats import chi2_contingency

   QUESTIONS = [
       # Five answerable from the scikit-learn docs
       "What is the default penalty for sklearn.linear_model.LogisticRegression?",
       "How does sklearn.cluster.KMeans choose initial centroids by default?",
       "What does sklearn.preprocessing.StandardScaler.fit_transform return?",
       "What scoring does sklearn.model_selection.cross_val_score use for a classifier by default?",
       "How do you set class weights in sklearn.svm.SVC?",
       # Five not in the docs (should refuse)
       "Does scikit-learn ship a transformer-based deep learning module?",
       "What is the official scikit-learn slack workspace URL?",
       "What is the scikit-learn maintainer team's payroll budget?",
       "Can I run scikit-learn natively on a TPU without a wrapper?",
       "What is the support phone number for scikit-learn?",
   ]

   REFUSAL_RE = re.compile(
       r"(don't have|do not have|cannot answer|not in (?:the|our) (?:documentation|catalog|docs)|"
       r"unable to|outside (?:the )?(?:scikit-learn|product catalog)|no information)",
       re.IGNORECASE,
   )

   def run_variant(branch: str) -> list[int]:
       subprocess.run(["git", "checkout", branch], check=True)
       refusals = []
       for q in QUESTIONS:
           r = requests.post(
               "http://localhost:8080/query",
               json={"question": q},
               timeout=30,
           )
           answer = r.json()["answer"]
           refusals.append(1 if REFUSAL_RE.search(answer) else 0)
       return refusals

   def main():
       a = run_variant("main")
       b = run_variant("ex3-soft-refusal")
       table = [
           [sum(a), len(a) - sum(a)],
           [sum(b), len(b) - sum(b)],
       ]
       chi2, p, _, _ = chi2_contingency(table)
       report = {
           "variant_a_refusals": sum(a),
           "variant_a_n": len(a),
           "variant_b_refusals": sum(b),
           "variant_b_n": len(b),
           "chi2": chi2,
           "p_value": p,
           "significant_at_0.05": p < 0.05,
       }
       json.dump(report, sys.stdout, indent=2)
       sys.stdout.write("\n")

   if __name__ == "__main__":
       main()
   ```

3. Run it. Make sure `make serve` is up. From this starter's root directory:

   ```
   uv run python scripts/ab_refusal.py
   ```

4. Read the report. It will print something like:

   ```json
   {
     "variant_a_refusals": 5,
     "variant_a_n": 10,
     "variant_b_refusals": 2,
     "variant_b_n": 10,
     "chi2": 1.978,
     "p_value": 0.160,
     "significant_at_0.05": false
   }
   ```

### Acceptance criterion

The script runs end-to-end without errors, produces a JSON report with both variants' refusal counts and a chi-squared p-value, and reports `significant_at_0.05` correctly. You do not need a statistically significant result — with N=10 per variant you almost certainly will not get one. The point is to run the analysis honestly and read out the conclusion: "not enough data to call it." That is the honest read in nine out of ten small-sample experiments, and it is the read most teams skip in favor of vibes.

In your commit message or a short markdown note, answer two questions. First: at this sample size, what would you need to see in production traffic — how many requests per variant — before the test could detect a twenty-percentage-point shift in refusal rate at ninety-five percent confidence? (Rule of thumb from the concept module is ten to twenty thousand requests per variant for a five percent relative shift in a binary metric. A twenty-percentage-point absolute shift is much larger, so the sample size is much smaller. Estimate it; do not look up the exact number.) Second: if variant B refuses fewer questions but the regex is heuristic, what would you do to confirm that the change is real and not a labeling artifact? (Acceptable answers include using an LLM-as-judge instead of a regex, having a human label a holdout sample, or measuring a downstream metric like thumbs-up rate that does not depend on labeling refusals.)

### Hints

<details>
<summary>If <code>chi2_contingency</code> complains about expected frequencies below 5</summary>

That is the standard caveat for the chi-squared approximation. With N=10 per variant and skewed refusal counts, your expected cells will often be below 5. The fix is either more data or switching to Fisher's exact test: `from scipy.stats import fisher_exact`. For the exercise, the warning is itself the lesson — small samples make the test unreliable.
</details>

<details>
<summary>If the refusal regex is over-matching or under-matching</summary>

The regex is a heuristic, not a classifier. In production you would use an LLM-as-judge or a labeled holdout set. For the exercise, eyeball the answers and adjust the pattern. Document which answers the regex disagreed with — that disagreement is the noise floor of your A/B metric.
</details>

<details>
<summary>If the script errors on <code>git checkout</code> because of uncommitted changes</summary>

You probably edited a file without committing. Either commit, stash, or revert before running. The script assumes clean checkouts between variants — that is part of the discipline of treating the prompt as a versioned artifact. A modified working tree at A/B time means the variant you tested is not the variant in any commit, which means you can not reproduce it.
</details>

## Common pitfalls

A few traps that catch most learners on this module. Skim before submitting:

- **`PROMPT_ENV` not set, or set in the wrong shell.** Exporting a variable in shell A does not affect shell B. If your loader returns the prod template when you expected dev, run `echo $PROMPT_ENV` in the same shell that runs pytest. If it is empty, `export PROMPT_ENV=dev` first, or use `PROMPT_ENV=dev uv run pytest …` to set it inline for one command. The inline form is preferred for tests because it is hermetic — no leftover state when the test is done.
- **Template syntax errors fail silently until rendered.** Jinja parses the template lazily. A missing `{% endif %}`, a mismatched `{{ }}`, or a typo in a variable name will not surface until `template.render()` is actually called. Run a quick render in a Python REPL before committing: `Environment(...).get_template(...).render(contexts="x")`. If the template uses variables you have not supplied, Jinja defaults the value to an empty string unless `StrictUndefined` is configured — which silences bugs you would rather catch. Consider adding `undefined=StrictUndefined` from `jinja2` to the Environment for exercise 2.
- **Sample size too small, interpreted as a positive result.** Five questions per variant is enough to develop intuition. It is not enough to declare a winner. If you find yourself writing "variant B is better" in your notes, ask whether the difference would survive at N=1000. Usually it would not. A useful self-check: simulate the outcome under the null hypothesis (no difference between variants) ten times and see how often you would have seen the observed split by pure chance. If it happens more than five times out of ten, your A/B is statistical noise.
- **Drift between dev and prod templates breaks tests.** If you edit `prompts/dev/docbot_system.j2` and forget the prod copy, your dev tests pass and your prod queries fail in surprising ways. A regression test that renders both templates against the same fixture catches this cheaply — assert that both renders contain the same set of grounding markers, or the same instruction count, or whatever invariants matter for your application. It is the unsexy part of prompt ops — write the test once and stop worrying.
- **Vocareum proxy timeouts on chained calls.** The Vocareum endpoint can rate-limit when the harness fires ten requests in a tight loop. If you see `429` or socket timeouts, add `time.sleep(0.5)` between requests, or run during off-peak hours. The proxy-side rate limit is real and not configurable from the learner side; the workaround is just to slow down. The starter's eval Makefile target defaults to single-worker concurrency for the same reason.
- **Forgetting to switch branches between A and B halves.** A common slip — you intended to test branch B but forgot to `git checkout`, so both halves measured the same prompt. The harness prints which branch it ran for each variant; check that the labels disagree before reading the p-value. The chi-squared test will happily compute a p-value on identical samples; the test does not know you forgot to switch branches.
- **Forgetting that the model itself drifts under you.** OpenAI ships new minor revisions of `gpt-4o` and `gpt-4o-mini` over time, and a small model-version change can shift response distributions independently of any prompt change. If your A/B straddles a model release, the variant difference is contaminated by the model difference. The fix is to pin the model alias in `src/config.py` to a versioned snapshot (e.g., `gpt-4o-2024-08-06`) for the duration of the test, and unpin afterward. The concept module mentioned this under reproducibility; this is where it actually bites.

## What you have now

Three skills, each tied to a starter file you can grep for. A template you can extend with variables and conditionals — that is the rendering half of versioning. An environment-aware loader that selects the right prompt at runtime — that is the runtime half. An A/B harness that gives you a real metric and an honest read on whether the difference is significant — that is the decision half. From here, every prompt change in the starter — and in every LLM application you build after this course — should pass through this same Git-then-evaluate loop.

Two skills you do not have yet, but will pick up in later modules. The eval gate that the concept module called out — a CI-time check that a prompt change does not regress quality below a threshold — uses RAGAS metrics and lands in Module 11. The richer monitoring window — watching latency, cost, and refusal rate over a defined post-deploy interval — uses Phoenix tracing and lands in Module 9. The pieces compose: this module gives you the artifact, Module 9 gives you the observation, Module 11 gives you the score. When you see them all in the same repo at the end of the course, that composition is the operational loop the concept module diagrammed.

Commit cleanly before moving on. The next module shifts to vector databases, where the artifact you version is the embedding configuration rather than the prompt — same MLOps discipline, different surface.
