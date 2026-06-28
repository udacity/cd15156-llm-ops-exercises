# Solution video script: Module 03

**Video (tracker):** Build a prompt loader and A/B test two versions
**Type:** Solution · **Target:** 3 to 5 minutes talking time · **Alignment:** Aligned, no tool drift
**Goal:** Walk one working solution to Exercises 1 to 3, showing it run.

> **Format:** **[SAY]** is narration to read. **[SHOW]** is the on-screen action. **[ACCELERATE]** marks long-running output the editor can speed up (it does not count toward talking time).
> Output tagged *(illustrative)* needs a live `OPENAI_API_KEY`, so capture it on the recording machine. The render checks and both test runs below were run for real while writing this script.
> **Key hints:** the shortcut in each `[SHOW]` is the fastest way to do that action on camera. VS Code defaults, `Ctrl` on Linux/Windows, `Cmd` on Mac. `Ctrl+P` opens a file by name, `Ctrl+G` jumps to a line, `` Ctrl+` `` toggles the terminal, `Ctrl+L` clears it.

---

### Intro

**[SAY]** This is one way to build the three pieces. You'll write a tiered template, a dev and prod loader, and an A/B test. Yours doesn't have to match mine, and if you get stuck, every block carries a TODO comment, so search the solution and jump to the code.

**[SAY]** Open a terminal and change into the solution directory.

**[SHOW]** (`` Ctrl+` `` opens the terminal)
```
cd solution
```

> **Production note:** environment setup (`make setup`, `.env` key, `make load-data`) lives in the Environment Setup doc and is not repeated here. Record from inside `solution/` with setup already done and the folder initialized as a Git repo with a configured identity (Exercise 3 uses branches and commits). Run `make setup` then `make load-data` in `solution/` so dependencies and the retrieval index are in place, and confirm `solution/.env` carries the three feature-flag lines from `.env.example` (`ENABLE_SEMANTIC_CACHE=false`, `ENABLE_OUTPUT_GUARD=false`, `TRACING_BACKEND=none`); a stale `.env` leaves the cache, output guard, and tracing on. Exercise 3 also needs the `ex3-soft-refusal` branch committed and the server running with `make serve`.

### Exercise 1: a tiered template

**[SAY]** Exercise 1 asks you to add a premium-tier note. But the real work is wiring one new variable, user_tier, through every layer in a single commit.

**[SHOW]** Open the template `prompts/docbot_system.j2` (`Ctrl+P` → `docbot_system`, then `Ctrl+G` → `26`), where an `if user_tier == "premium"` block adds instruction 8.
```jinja
{% if user_tier == "premium" %}
8. **Premium-tier note.** ... scikit-learn-help mailing list ...
{% endif %}
```

**[SAY]** Next the renderer passes that variable in. The decision worth calling out is to default it to "standard", so existing callers keep working.

**[SHOW]** `src/generator.py` (`Ctrl+P` → `generator`):
```python
def render_system_prompt(sources, user_tier: str = "standard") -> str:
    ...
    return template.render(contexts=contexts, user_tier=user_tier)
```

**[SAY]** This is the easy place to slip. Change the template but not the signature and you get a TypeError. Forget to pass it in anywhere and there's no error at all, it just stays standard forever. Wiring every layer in one commit prevents that silent bug.

**[SHOW]** The test covers both tiers. *(real output)*
```
uv run pytest tests/test_prompt_tier.py -q        # 2 passed
```

**[SAY]** Premium renders instruction 8, and standard leaves it out.

### Exercise 2: a dev and prod loader

**[SAY]** Exercise 2 splits the prompt into two versions and loads the right one at runtime. You make two folders, dev and prod; the dev copy adds a debug line that tags its answers.

**[SHOW]** `prompts/dev/docbot_system.j2` next to `prompts/prod/docbot_system.j2` (`Ctrl+P` → `dev/docbot`, then `prod/docbot`). Only dev carries the `[DEV]` suffix.

**[SAY]** Then you write the loader. It reads PROMPT_ENV and returns a Jinja environment pointed at that folder, and it defaults to prod, not dev.

**[SHOW]** `src/prompts/loader.py` (`Ctrl+P` → `loader`):
```python
def load_environment(env_name=None):
    env_name = env_name or os.environ.get("PROMPT_ENV", "prod")
    ...
```

**[SAY]** Defaulting to prod is deliberate. A dev prompt slipping into production is exactly the failure this guards against, so an unrecognized value raises a `ValueError` rather than guessing.

**[SAY]** Two things trip people up: the loader has to walk up to the project root to find the prompts folders, and the package needs an empty __init__.py or the import fails. Also, PROMPT_ENV has to be set in the same shell that runs your tests.

**[SHOW]** *(real output)*
```
uv run pytest tests/test_prompt_loader.py -q      # 4 passed
```

**[SAY]** Prod loads by default, dev injects the marker, an explicit argument wins over the variable, and an unknown value raises a `ValueError`.

### Exercise 3: an A/B test

**[SAY]** Exercise 3 is where versioning starts to pay off. You create a second prompt on its own branch, then compare the two with a real metric instead of a hunch.

**[SHOW]** Switch to the variant branch you created in Exercise 3:
```
git checkout ex3-soft-refusal
```

**[SHOW]** Open `prompts/docbot_system.j2` (`Ctrl+P` → `docbot_system`, then `Ctrl+G` → `17`). On this branch, instruction 3 leans toward answering instead of being honest about uncertainty.

**[SAY]** Then you build the harness: it asks the same ten questions to each branch, five answerable and five out of scope, labels each answer as a refusal or not, and runs a chi-squared test.

**[SHOW]** `scripts/ab_refusal.py` (`Ctrl+P` → `ab_refusal`): the ten `QUESTIONS`, the `REFUSAL_RE` that labels answers, `run_variant` checking out each branch, and `chi2_contingency` at the end.

**[SAY]** One thing the harness assumes is a clean working tree at each checkout. If you've edited a prompt without committing, the `git checkout` fails, which is the harness reminding you that the variant you're testing has to be a real commit.

**[SHOW][ACCELERATE]** *(illustrative, needs a key and a running server)* Run the harness. It checks out each branch, asks all ten questions, and prints the report:
```
make serve
uv run python scripts/ab_refusal.py
```
A representative run, your exact numbers will vary:
```json
{
  "variant_a_refusals": 3, "variant_a_n": 10,
  "variant_b_refusals": 3, "variant_b_n": 10,
  "chi2": 0.0, "p_value": 1.0, "significant_at_0.05": false
}
```

**[SAY]** Read the JSON, don't skim it. The two refusal counts are out of ten each, and the p-value with significant_at_0.05 tell you whether a difference is real or just noise. At ten per side the counts often come back equal or close, so the p-value sits high and there's no difference you can trust. That's a real result, not a failure, so write it down instead of fishing for a win.

**[SAY]** But don't read that as the prompt doing nothing. The counts matched; the answers didn't. Look at the same question on each branch.

**[SHOW]** *(illustrative, needs a key)* Same question on the variant branch:
```
git checkout ex3-soft-refusal
curl -s -X POST http://localhost:8080/query -H 'Content-Type: application/json' \
  -d '{"question": "Does scikit-learn ship a transformer-based deep learning module?"}' | uv run python -c "import sys, json; print(json.load(sys.stdin)['answer'])"
```
It answers directly: "No, scikit-learn does not ship a transformer-based deep learning module."

**[SHOW]** *(illustrative, needs a key)* And on the baseline:
```
git checkout main
curl -s -X POST http://localhost:8080/query -H 'Content-Type: application/json' \
  -d '{"question": "Does scikit-learn ship a transformer-based deep learning module?"}' | uv run python -c "import sys, json; print(json.load(sys.stdin)['answer'])"
```
It hedges first: "Based on the documentation excerpts retrieved, scikit-learn does not offer..."

**[SAY]** Same question, two different answers, and the refusal count couldn't tell them apart. The prompt change was real. Our metric was just too coarse to see it.

**[SAY]** Two questions close the exercise: how many requests per variant you'd need to trust a shift this size, and how you'd confirm a real difference when the labels come from a rough regex.

### Close

**[SAY]** That's one way through all three exercises. You wired a new variable through every layer, built a loader that's safe by default, and ran an A/B test with a real metric. If a piece tripped you up, search the solution for its TODO comment and you'll land right on the code.
