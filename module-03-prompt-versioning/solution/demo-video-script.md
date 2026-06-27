# Demo video script: Module 03

**Video (tracker):** Set up Git-backed prompt templates (dev/prod)
**Type:** Demo · **Target:** 3 to 5 minutes talking time · **Alignment:** Aligned, no tool drift
**Goal:** Orient students to the tools they will use for prompt versioning, and show the moving parts running, before they attempt the exercises.

> **Format:** **[SAY]** is narration to read. **[SHOW]** is the on-screen action. **[ACCELERATE]** marks long-running output the editor can speed up (it does not count toward talking time).
> Output tagged *(illustrative)* needs a live `OPENAI_API_KEY`, so capture it on the recording machine. Everything else was run for real while writing this script.
> **Key hints:** the shortcut in each `[SHOW]` is the fastest way to do that action on camera. VS Code defaults, `Ctrl` on Linux/Windows, `Cmd` on Mac. `Ctrl+P` opens a file by name, `Ctrl+G` jumps to a line, `` Ctrl+` `` toggles the terminal, `Ctrl+L` clears it.
> **Production note** blocks are recorder guidance (workspace quirks, setup), not spoken lines, so they do not count toward talking time.

---

### Before you start

**[SAY]** This module is about treating a prompt like versioned code. In the exercises you'll build three things: prompt templates kept in Git, a loader that picks the right version per environment, and a small A/B test to compare two versions. This demo shows the tools you will be using for these exercises.

> **Production note:** environment setup (install, `.env` key, `make load-data`) is covered in the Environment Setup doc, so it is not repeated here. Record from inside `prompt-versioning-starter/` with that setup already done, and with the folder initialized as a Git repo with a configured identity (the demo and exercises use Git branches and commits). The `/query` calls below need a live `OPENAI_API_KEY`. Before recording, confirm this starter's `.env` carries the three feature-flag lines from `.env.example`: `ENABLE_SEMANTIC_CACHE=false`, `ENABLE_OUTPUT_GUARD=false`, `TRACING_BACKEND=none`. A live `.env` is gitignored and is not updated by changes to `.env.example`, so a stale one leaves the cache, output guard, and tracing on, which masks this lesson (the output guard will refuse the `/query` answers). Restart the server after editing `.env`, since settings load once at startup.

### 1. A prompt is a file under version control

**[SAY]** Here's a prompt template. It's a plain text file with a Jinja2 placeholder, the double-brace `contexts`, where the retrieved docs get slotted in when the prompt is rendered. Keeping the prompt in a file is what lets Git track every version of it, the same as any code.

**[SHOW]** Open `prompts/docbot_system.j2` (`Ctrl+P` → type `docbot_system`), scroll the seven instructions, and point at the placeholder:
```
<<<BEGIN_CONTEXT>>>
{{ contexts }}
<<<END_CONTEXT>>>
```

### 2. Render it, then serve it

**[SAY]** Rendering turns the template plus the retrieved chunks into the actual system message. You can call it on its own, with no model involved, which makes it easy to test.

**[SHOW]** *(real output)*
```
uv run python -c "from src.generator import render_system_prompt; from src.models import Source; print(render_system_prompt([Source(doc_id='x', chunk_text='LogisticRegression default penalty is l2.', similarity_score=0.9)])[:1000])"
```
It prints the rendered prompt, with the chunk sitting inside the context block.

**[SAY]** Now let's serve the gateway and hit the live API from the terminal.

**[SHOW]** Start the server:
```
make serve
```

**[SHOW]** *(illustrative, needs a key)* Send the query from the terminal:
```
curl -s -X POST http://localhost:8080/query -H 'Content-Type: application/json' \
  -d '{"question": "What is the default penalty for sklearn.linear_model.LogisticRegression?"}' | uv run python -c "import sys, json; print(json.load(sys.stdin)['answer'])"
```
It prints a grounded answer that cites the API.

### 3. Switching versions changes the answer

**[SAY]** This is the payoff, and it's why we commit. With the variant saved as a commit on its own branch, I can switch prompt versions with a Git checkout and ask the exact same question. Same retrieval, same code; the only thing that changes is which committed version Git puts on disk, and the answer follows it.

**[SHOW]** Create a branch for the variant:
```
git checkout -b prompt-banner
```

**[SHOW]** In `prompts/docbot_system.j2` (`Ctrl+P` → `docbot_system`, then `Ctrl+G` → `19`), replace instruction 4 with: **Begin every answer with the exact prefix `[PROMPT v2] ` (keep the trailing space).** Save the file (`Ctrl+S`).

**[SHOW]** Verify the edit is saved before committing (an empty diff means the file was not saved):
```
git diff -- prompts/docbot_system.j2
```

**[SHOW]** Commit it, so the variant becomes a switchable version:
```
git commit -am "banner prompt variant"
```

**[SAY]** Now the comparison. I'll ask the exact same question on each committed version and watch the answer follow whichever prompt Git has on disk.

**[SHOW]** Switch to the baseline version:
```
git checkout main
```

**[SHOW][ACCELERATE]** *(illustrative, needs a key)* Ask the question; the baseline answer has no prefix:
```
curl -s -X POST http://localhost:8080/query -H 'Content-Type: application/json' \
  -d '{"question": "What is the default penalty for sklearn.linear_model.LogisticRegression?"}' | uv run python -c "import sys, json; print(json.load(sys.stdin)['answer'])"
```

**[SHOW]** Switch to the banner version:
```
git checkout prompt-banner
```

**[SHOW][ACCELERATE]** *(illustrative, needs a key)* Ask the exact same question; this version's answer now starts with the `[PROMPT v2]` banner:
```
curl -s -X POST http://localhost:8080/query -H 'Content-Type: application/json' \
  -d '{"question": "What is the default penalty for sklearn.linear_model.LogisticRegression?"}' | uv run python -c "import sys, json; print(json.load(sys.stdin)['answer'])"
```

**[SAY]** Same question both times. The answer changed only because Git checked out a different committed version of the prompt. That is prompt versioning: each version is a commit you can switch to, diff, tag, or roll back.

**[SHOW]** Back to the baseline and confirm the history:
```
git checkout main
git log --oneline -- prompts/
```

> **Production note:** if an answer does not change after a `git checkout`, restart `make serve` so it re-reads the prompt from disk (its `--reload` watches `src/`, not `prompts/`).

### Recap and what's next

**[SAY]** So that's the toolkit: templates rendered by Jinja2, versioned by Git, served behind one endpoint. In the exercises you'll formalize it into a tiered template, a dev and prod loader, and a real A/B test. If you get stuck, you can check out the solution video to see how I did it. See you in the next module!
