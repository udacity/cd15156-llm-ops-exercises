# Solution video script: Module 03

**Video (tracker):** Build a prompt loader and A/B test two versions
**Type:** Solution · **Target:** 3 to 5 minutes talking time · **Alignment:** Aligned, no tool drift
**Goal:** Walk one working solution to Exercises 1 to 3, showing it run.

> **Format:** **[SAY]** is narration to read. **[SHOW]** is the on-screen action. **[ACCELERATE]** marks long-running output the editor can speed up (it does not count toward talking time).
> Output tagged *(illustrative)* needs a live `OPENAI_API_KEY`, so capture it on the recording machine. The render checks and both test runs below were run for real while writing this script.
> **Key hints:** the shortcut in each `[SHOW]` is the fastest way to do that action on camera. VS Code defaults, `Ctrl` on Linux/Windows, `Cmd` on Mac. `Ctrl+P` opens a file by name, `Ctrl+G` jumps to a line, `` Ctrl+` `` toggles the terminal, `Ctrl+L` clears it.

---

### Intro

**[SAY]** This is one way to build the three pieces: the tiered template, the dev and prod loader, and the A/B test. Yours doesn't have to match mine. And if you get stuck, every block carries a TODO comment, so search the solution and jump to the code.

**[SAY]** Open a terminal and change into the starter directory.

**[SHOW]** (`` Ctrl+` `` opens the terminal)
```
cd prompt-versioning-starter
```

### Exercise 1: a tiered template

**[SAY]** Exercise 1 asks you to add a premium-tier note. The real work isn't the note, it's wiring one new variable, user_tier, through every layer in a single commit.

**[SHOW]** Open the template `prompts/docbot_system.j2` (`Ctrl+P` → `docbot_system`, then `Ctrl+G` → `26`), where an `if user_tier == "premium"` block adds instruction 8.
```jinja
{% if user_tier == "premium" %}
8. **Premium-tier note.** ... scikit-learn-help mailing list ...
{% endif %}
```

**[SAY]** Next the renderer passes that variable in, and here's a decision you make: default it to "standard" so existing callers keep working.

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

**[SAY]** Then you write the loader: it reads PROMPT_ENV and returns a Jinja environment pointed at that folder. The decision that matters: default to prod, not dev.

**[SHOW]** `src/prompts/loader.py` (`Ctrl+P` → `loader`):
```python
def load_environment(env_name=None):
    env_name = env_name or os.environ.get("PROMPT_ENV", "prod")
    ...
```

**[SAY]** Defaulting to prod is deliberate: a dev prompt slipping into production is exactly the failure this guards against, so an unexpected value raises instead of guessing.

**[SAY]** Two things trip people up: the path math has to climb from the loader file back to the project root, and the package needs an empty __init__.py or the import fails. Also, PROMPT_ENV has to be set in the same shell that runs your tests.

**[SHOW]** *(real output)*
```
uv run pytest tests/test_prompt_loader.py -q      # 4 passed
```

**[SAY]** Prod loads by default, dev injects the marker, an explicit argument wins over the variable, and a bad value raises.

### Exercise 3: an A/B test

**[SAY]** Exercise 3 is where versioning meets measurement. You create a second prompt on its own branch, then compare the two with a real metric instead of a hunch.

**[SHOW]** Open `prompts/docbot_system.j2` on the `ex3-soft-refusal` branch (`Ctrl+P` → `docbot_system`), where instruction 3 is rewritten to lean toward answering.

**[SAY]** Then you build the harness: it asks the same ten questions to each branch, five answerable and five out of scope, labels each answer as a refusal or not, and runs a chi-squared test.

**[SHOW]** `scripts/ab_refusal.py` (`Ctrl+P` → `ab_refusal`): the ten `QUESTIONS`, the `REFUSAL_RE` that labels answers, `run_variant` checking out each branch, and `chi2_contingency` at the end.

**[SAY]** One thing the harness assumes is a clean working tree at each checkout. If you've edited a prompt without committing, the `git checkout` fails, which is the harness reminding you that the variant you're testing has to be a real commit.

**[SHOW][ACCELERATE]** *(illustrative, needs a key and a running server)*
```
make serve
uv run python scripts/ab_refusal.py        # prints JSON with chi2 and p_value
```

**[SAY]** With ten questions a side you'll usually see a gap but no statistically significant result, and that's the expected outcome. The exercise ends with two questions: how many requests per variant you'd really need to trust a shift this size, and how you'd confirm a real difference when the labels come from a rough regex.

### Close

**[SAY]** That's one way through all three: a variable wired cleanly through every layer, a loader that defaults to the safe environment, and an A/B test you can read. If a piece tripped you up, search the solution for its TODO comment and you'll land on the answer.
