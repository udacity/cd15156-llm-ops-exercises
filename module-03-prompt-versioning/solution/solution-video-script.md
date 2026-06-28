# Solution video script: Module 03

**Video (tracker):** Build a prompt loader and A/B test two versions
**Type:** Solution · **Target:** 3 to 5 minutes talking time · **Alignment:** Aligned, no tool drift
**Goal:** Walk one working solution to Exercises 1 to 3, showing it run.

> **Format:** **[SAY]** is narration to read. **[SHOW]** is the on-screen action. **[ACCELERATE]** marks long-running output the editor can speed up (it does not count toward talking time).
> **Flow:** every file beat opens the file first in a **[SHOW]**, then you talk about it in the **[SAY]** that follows. Each post-open **[SAY]** starts with a short transition ("Start in the template", "Now the renderer", "Over in the loader") so the move between files is smooth and you are never talking about a file that is not on screen yet.
> Output tagged *(real output)* was run while writing this script. *(illustrative)* needs a live `OPENAI_API_KEY`; *(pre-captured)* is real output recorded earlier, shown on screen with no live call.
> **Key hints:** `Ctrl` on Linux/Windows, `Cmd` on Mac. `Ctrl+P` opens a file by name, `Ctrl+G` jumps to a line, `` Ctrl+` `` toggles the terminal, `Ctrl+L` clears it.

---

### Intro

**[SAY]** This is one way to build the three pieces: a tiered template, a dev and prod loader, and an A/B test. Yours doesn't have to match mine, and every block carries a TODO comment, so if you get stuck, grep the solution and jump to the code.

**[SHOW]** Open the terminal (`` Ctrl+` ``) and change into the solution directory:
```
cd solution
```

> **Production note:** environment setup (`make setup`, `.env` key, `make load-data`) lives in the Environment Setup doc and is not repeated here. Record from inside `solution/` with setup already done and the folder initialized as a Git repo with a configured identity (Exercise 3 uses branches and commits). Run `make setup` then `make load-data` in `solution/` so dependencies and the retrieval index are in place, and confirm `solution/.env` carries the three feature-flag lines from `.env.example` (`ENABLE_SEMANTIC_CACHE=false`, `ENABLE_OUTPUT_GUARD=false`, `TRACING_BACKEND=none`); a stale `.env` leaves the cache, output guard, and tracing on. Exercise 3 also needs the `ex3-soft-refusal` branch committed and the server running with `make serve`.

### Exercise 1: a tiered template

**[SAY]** Exercise 1 adds a premium-tier note. The real work isn't the note, it's wiring one new variable, user_tier, through every layer in one commit.

**[SHOW]** Open `prompts/docbot_system.j2` (`Ctrl+P` → `docbot_system`, then `Ctrl+G` → `26`).
**[SAY]** Start in the template. At instruction 8, an `if user_tier == "premium"` block adds the note only for premium users.

**[SHOW]** Open `src/generator.py` (`Ctrl+P` → `generator`).
**[SAY]** Now the renderer. It takes user_tier and defaults to "standard", so existing callers keep working. That default is the one real decision. Miss the signature and you get a TypeError; forget to pass it through and it silently stays standard.

**[SHOW]** Back in the terminal, run the tier test: *(real output)*
```
uv run pytest tests/test_prompt_tier.py -q        # 2 passed
```
**[SAY]** Premium renders instruction 8, standard leaves it out.

### Exercise 2: a dev and prod loader

**[SAY]** Exercise 2 splits the prompt into two versions and loads the right one at runtime.

**[SHOW]** Open `prompts/dev/docbot_system.j2`, then `prompts/prod/docbot_system.j2` (`Ctrl+P` → `dev/docbot`, then `prod/docbot`).
**[SAY]** Two folders, dev and prod. Only the dev copy carries the `[DEV]` suffix that tags its answers.

**[SHOW]** Open `src/prompts/loader.py` (`Ctrl+P` → `loader`).
**[SAY]** Over in the loader. It reads PROMPT_ENV and returns a Jinja environment for that folder, defaulting to prod. That's deliberate: a dev prompt reaching production is the failure this guards against, so a bad value raises a ValueError instead of guessing. The gotcha to watch: the package needs an empty __init__.py or the import fails.

**[SHOW]** Back in the terminal, run the loader test: *(real output)*
```
uv run pytest tests/test_prompt_loader.py -q      # 4 passed
```
**[SAY]** Prod loads by default, dev injects the marker, an explicit argument wins, and a bad value raises.

### Exercise 3: an A/B test

**[SAY]** Exercise 3 is where versioning pays off. You put a second prompt on its own branch and compare the two with a real metric.

**[SHOW]** In the terminal, switch to the variant branch:
```
git checkout ex3-soft-refusal
```
**[SHOW]** Open `prompts/docbot_system.j2` (`Ctrl+P` → `docbot_system`, then `Ctrl+G` → `17`).
**[SAY]** On this branch, instruction 3 leans toward answering instead of staying honest about uncertainty. That one line is the only difference between the prompts.

**[SHOW]** Open `scripts/ab_refusal.py` (`Ctrl+P` → `ab_refusal`).
**[SAY]** Now the harness. It asks the same ten questions on each branch, labels each answer a refusal or not with a regex, and runs a chi-squared test. It checks out a branch per variant, so it needs a clean working tree.

**[SHOW][ACCELERATE]** Back in the terminal, run it: *(illustrative, needs a key and a running server)*
```
make serve
uv run python scripts/ab_refusal.py
```
Example only, your counts will differ run to run; read your own off the screen:
```json
{ "variant_a_refusals": 3, "variant_b_refusals": 2, "chi2": 0.0, "p_value": 1.0, "significant_at_0.05": false }
```
**[SAY]** Read the JSON, don't skim it. The counts are refusals out of ten. Whatever the split, a tie or off by one or two, the p-value stays high and significant-at-0.05 is false. No difference you can trust yet, and that's a real result, so write it down instead of fishing for a win.

**[SAY]** But the prompt didn't do nothing; the metric just barely moved. Here's the same question answered on each branch:
**[SHOW]** *(pre-captured, no live call)*
```
variant  (ex3-soft-refusal): "No, scikit-learn does not ship a transformer-based deep learning module..."
baseline (main):             "Based on the documentation excerpts retrieved, scikit-learn does not offer..."
```
**[SAY]** Same facts, different framing. The variant answers straight; the baseline hedges first. The count barely registered that, but the change was real. The metric was just too coarse to catch it.

**[SAY]** The exercise closes with two questions: how many requests per variant you'd need to trust a shift this size, and how you'd confirm a real difference when your labels come from a rough regex.

### Close

**[SAY]** That's one way through all three: a variable wired through every layer, a loader that's safe by default, and an A/B test you can read. If a piece tripped you up, grep the solution for its TODO comment and you'll land on the code.
