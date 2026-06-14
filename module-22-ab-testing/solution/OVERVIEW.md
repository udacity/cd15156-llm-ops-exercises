---
module_number: 22
module_title: "Conduct Prompt A/B Tests with Python Feature Flags"
slug: ab-testing
unit_type: overview-video-script
target_minutes: "5-7"
word_count: 719
---

# Module 22 Overview Video: Prompt A/B Testing with Feature Flags

> Recording aid for a 5-7 min overview (read pace ~100 wpm). Read all prose aloud, top to bottom. `*[Stage: ...]*` lines are screen directions, not spoken. Not learner-facing.

## 0. VS Code setup (before you hit record)

Open the files you will show on camera (run from the repo root):

```bash
code exercises/22-ab-testing/ab-testing-starter/DEMO.md exercises/22-ab-testing/ab-testing-starter/INSTRUCTIONS.md exercises/22-ab-testing/ab-testing-starter/INTERFACES.md exercises/22-ab-testing/ab-testing-starter/src/optimization/ab.py exercises/22-ab-testing/ab-testing-starter/prompts/docbot_system_A.j2 exercises/22-ab-testing/ab-testing-starter/prompts/docbot_system_B.j2 exercises/22-ab-testing/ab-testing-starter/prompts/judge.j2 exercises/22-ab-testing/ab-testing-starter/scripts/ab_analyze.py
```

Files and why each is on screen:
- `DEMO.md`: the walkthrough you will reference; shows the three A/B primitives end to end.
- `INSTRUCTIONS.md`: the three exercises; keep it open to point at each acceptance criterion.
- `INTERFACES.md`: the frozen signatures the starter depends on; useful for the retrieval seam the harness reuses.
- `src/optimization/ab.py`: the three primitives, `pick_variant`, `call_with_variant`, and `log_assignment`. The sticky hash lives here.
- `prompts/docbot_system_A.j2`: variant A, the "be concise and direct" prompt.
- `prompts/docbot_system_B.j2`: variant B, identical except instruction four says "be expansive." The one-line diff is the point.
- `prompts/judge.j2`: the LLM-as-judge faithfulness prompt the harness scores each answer against.
- `scripts/ab_analyze.py`: the analyzer behind `make ab-analyze`; it builds the table and runs the chi-square test.

## 1. Intro  (~45-60s)

*[Stage: DEMO.md on screen.]*

In this module you'll run a real A/B test on two prompt variants. A/B testing means splitting traffic between two versions and measuring which one wins. You'll route each call with a feature flag, score every answer, and read the result honestly.

There's a catch that makes large language models, or LLMs, different. The two variants can come out tied, and tied is often the truth, not a bug. The real skill is knowing what the numbers can and can't tell you.

## 2. Topic overview  (~60-90s)

*[Stage: open src/optimization/ab.py, point to the pick_variant function.]*

The starter ships three small pieces. A picker decides which variant a call gets. A router renders that variant's prompt and calls the model. A logger writes one row per call for later analysis.

The key idea is sticky assignment. The picker hashes the client's identity, so the same client lands on the same variant every call. If a user bounced between variants, you couldn't tell a prompt effect from a who-saw-what effect. Sticky keeps each user on one side, so the comparison stays fair.

Here's the misconception to clear up now. More calls doesn't always mean more evidence. When every user is stuck on one variant, their calls are correlated, not independent. So what counts for the test isn't the call count. It's the number of unique users.

## 3. Exercise call-outs

### Exercise 1: Write the harness and run 200 simulated calls

*[Stage: switch to INSTRUCTIONS.md, Exercise 1; then point to judge.j2.]*

The first exercise wires the three primitives into one script. It fires two hundred calls across a pool of fifty clients, and scores each answer for success.

Start with the concept. You might reach for a simple check: did the answer cite a source id? On this corpus that check always fails. The ids are section anchors the model never repeats word for word. So every call would score false and the test would collapse. That's why the exercise scores success with an LLM-as-judge instead: it reads the answer against the retrieved text and gives an honest verdict.

Watch out for one thing above all. Keep the model on the cheap one, and always pass a client id. Drop the client id and the picker falls back to random, stickiness breaks, and your numbers are compromised. You're done when the stickiness check reports zero multi-variant clients.

### Exercise 2: Analyze with `make ab-analyze` and read the result honestly

*[Stage: scroll to Exercise 2; point to scripts/ab_analyze.py.]*

The second exercise runs the analyzer. It builds a two-by-two table of variant against success and runs a chi-square test, which asks whether a gap this size could happen by chance.

Here's the heart of the module. At this sample size the two variants will almost certainly come out tied, with a high p-value. A high p-value means the gap you see could easily be random noise. The honest read is "we can't tell these apart yet," not "they're the same." Those are different claims, and tied is the correct result here, not something to fix.

There's a sticky wrinkle too. The analyzer prints unique clients, around fifty, next to the two hundred calls. The fifty is what really powers the test. You're done when your written read names both the p-value and that effective sample size.

### Exercise 3: Cost and latency comparison, plus a written decision

*[Stage: scroll to Exercise 3; point to the per-variant table in the output.]*

The third exercise looks past quality. Two variants can tie on quality and still differ on cost and latency. Either one can decide the winner.

Here's why. Variant B asks for longer answers. Output tokens cost several times what input tokens do, so a wordier prompt quietly doubles the per-call bill, and it's slower too. At a tie on quality, the cheaper, faster variant wins.

So the watch-out is simple: don't stop at the p-value. Read the cost and latency rows too. You're done when you've written a short decision that names the winner, the metric it rests on, and how confident you are. "We need more data" is a valid decision, as long as you say so.

## 4. Key insights  (~30-45s)

*[Stage: return to src/optimization/ab.py, point to the salt parameter in pick_variant.]*

Three takeaways. First, sticky assignment buys a fair comparison, but it costs you statistical power, because what counts is unique users, not raw calls. Second, a tie is a real finding: "no evidence of a difference" is not the same as "no difference." Third, quality isn't the only metric. Cost and latency often decide the winner when quality ties.

## 5. Outro  (~20-30s)

*[Stage: return to DEMO.md.]*

That's an A/B test on prompts: a sticky flag, a judge for scoring, and a chi-square read you can trust. In production you'd swap the in-process flag for a managed one and run longer for real power. The skeleton stays the same. See you in the next one.
