---
module_number: 3
module_title: "Implement a Prompt Versioning System with Git and Jinja2"
slug: prompt-versioning
unit_type: overview-video-script
target_minutes: "5-7"
word_count: 695
---

# Module 03 Overview Video: Prompt Versioning with Git and Jinja2

> Recording aid for a 5-7 min overview (read pace ~100 wpm). Read all prose aloud, top to bottom. `*[Stage: ...]*` lines are screen directions, not spoken. Not learner-facing.

## 0. VS Code setup (before you hit record)

Open the files you will show on camera (run from the repo root):

```bash
code exercises/03-prompt-versioning/prompt-versioning-starter/DEMO.md exercises/03-prompt-versioning/prompt-versioning-starter/INSTRUCTIONS.md exercises/03-prompt-versioning/prompt-versioning-starter/INTERFACES.md exercises/03-prompt-versioning/prompt-versioning-starter/prompts/docbot_system.j2 exercises/03-prompt-versioning/prompt-versioning-starter/src/generator.py exercises/03-prompt-versioning/prompt-versioning-starter/src/models.py exercises/03-prompt-versioning/prompt-versioning-starter/src/prompts/loader.py exercises/03-prompt-versioning/prompt-versioning-starter/scripts/ab_refusal.py
```

Files and why each is on screen:
- `DEMO.md`: the walkthrough you will reference; it builds the loader from a stub and runs a tiny version switch end to end.
- `INSTRUCTIONS.md`: the three exercises; keep it open to point at each acceptance criterion.
- `INTERFACES.md`: the frozen signatures, so the learner sees what "do not change the signature" means for `render_system_prompt` and `generate`.
- `prompts/docbot_system.j2`: the prompt template; this is the artifact you version, and the file Exercise 1 extends.
- `src/generator.py`: the rendering code that loads the template and puts it on the wire to the model.
- `src/models.py`: the `Source` model (`doc_id`, `chunk_text`, `similarity_score`) so the render input is concrete.
- `src/prompts/loader.py`: the stub Exercise 2 fills in; the environment-aware loader.
- `scripts/ab_refusal.py`: the stub Exercise 3 fills in; the A/B refusal harness.

## 1. Intro  (~45-60s)

*[Stage: DEMO.md on screen.]*

In this module you'll treat a prompt the way you already treat code: as something you version, branch, and review. The prompt here is a real file that drives a documentation chatbot, and you'll learn to change it safely instead of editing a string buried in your application.

By the end you'll have extended a prompt template, built a loader that picks the right version at runtime, and run a small experiment to compare two versions head to head. The skills are small, but they're the foundation every later module builds on.

## 2. Topic overview  (~60-90s)

*[Stage: bring prompts/docbot_system.j2 forward, then point at src/generator.py.]*

Here's the core idea. A prompt is a versioned artifact, something you manage like code, with Git for history and Jinja2 templating for the parts that change per request. It isn't a string pasted into a Python file. It lives in its own file, on disk, where you can diff it, review it, and roll it back.

Jinja2 is a templating tool. It lets you leave a blank in the template, like the retrieved documentation, and fill that blank in at the moment of each request.

*[Stage: point at src/generator.py, the render then call sequence.]*

Now the part that surprises people. Switching prompt versions is a Git operation, not a code change. You check out a different branch or tag, and the loader reads whatever version is on disk and puts that selected version on the wire to the model. The application code never changes. That separation between the prompt and the code is the whole point.

## 3. Exercise call-outs

### Exercise 1: Extend the system template

*[Stage: switch to INSTRUCTIONS.md, Exercise 1; point at prompts/docbot_system.j2.]*

The first exercise adds one new variable to the template: a tier flag that injects an extra line for premium users. Simple as it sounds, it forces you to touch the three places a prompt change really lives: the template file, the code that renders it, and a test.

Here's what to watch out for. The most common silent failure in prompt work is wiring a new variable in one place and forgetting the others. The template renders, the test passes, and the variable is just always its default because nothing ever set it. The discipline is to add it at every layer in one commit. You're done when the test passes for both the premium and the standard tier.

### Exercise 2: Environment-aware prompt loader

*[Stage: open src/prompts/loader.py, the stub you'll fill in.]*

The second exercise builds the runtime half of versioning: a loader that reads an environment variable and picks a development or a production copy of the prompt. One application, two configurations, no conditionals scattered through your code.

Here's the watch-out. The default has to be production, on purpose. Accidentally promoting development behavior into production is exactly the bug this guards against. So an unset variable must land on prod, an explicit choice must win over the environment, and a bad value must raise an error rather than guess. You're done when all four cases pass.

### Exercise 3: A/B test with a real metric and a significance check

*[Stage: open scripts/ab_refusal.py; keep INSTRUCTIONS.md Exercise 3 in view.]*

The third exercise is the real one. An A/B test means running two versions side by side and comparing them on a number, here the rate at which the bot refuses to answer. You'll run ten questions through each version and check whether the gap is statistically significant, which just means: is this difference real, or could it be chance?

Here's the lesson, and it's the most important thing to say. At ten questions per version, the test almost certainly will not reach significance. That is the expected result, not a failure. The honest read is "not enough data to call it," and writing that down is the skill. You're done when the harness runs clean and reports its result truthfully, even when that result is uncertain.

## 4. Key insights  (~30-45s)

*[Stage: return to prompts/docbot_system.j2 next to src/generator.py.]*

Three takeaways. First, a prompt is an artifact you version with Git, not a string you bury in code. Second, changing versions is a Git operation, and the loader is what puts the selected version on the wire. Third, a small experiment tells you a direction, but only enough samples tell you whether it's real, so report the uncertainty instead of hiding it.

## 5. Outro  (~20-30s)

*[Stage: return to DEMO.md.]*

That's prompt versioning: a template on disk, a loader that selects a version, and an honest A/B to compare them. The next module moves to vector databases, where the artifact you version is the embedding configuration instead of the prompt. Same discipline, different surface. See you in the next one.
