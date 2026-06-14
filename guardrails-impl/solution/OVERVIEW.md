---
module_number: 20
module_title: "Implement Input/Output Guardrails with LLM Guard"
slug: guardrails-impl
unit_type: overview-video-script
target_minutes: "5-7"
word_count: 718
---

# Module 20 Overview Video: Guardrails Implementation

> Recording aid for a 5-7 min overview (read pace ~100 wpm). Read all prose aloud, top to bottom. `*[Stage: ...]*` lines are screen directions, not spoken. Not learner-facing.

## 0. VS Code setup (before you hit record)

Open the files you will show on camera (run from the repo root):

```bash
code exercises/20-guardrails-impl/guardrails-impl-starter/DEMO.md exercises/20-guardrails-impl/guardrails-impl-starter/INSTRUCTIONS.md exercises/20-guardrails-impl/guardrails-impl-starter/INTERFACES.md exercises/20-guardrails-impl/guardrails-impl-starter/src/guardrails/input_guards.py exercises/20-guardrails-impl/guardrails-impl-starter/src/guardrails/llm_guard/input_guards.py exercises/20-guardrails-impl/guardrails-impl-starter/src/guardrails/llm_judge/output_guards.py exercises/20-guardrails-impl/guardrails-impl-starter/src/guardrails/rate_limit.py exercises/20-guardrails-impl/guardrails-impl-starter/src/gateway/routes.py exercises/20-guardrails-impl/guardrails-impl-starter/src/generator.py exercises/20-guardrails-impl/guardrails-impl-starter/src/models.py exercises/20-guardrails-impl/guardrails-impl-starter/prompts/judge.j2
```

Files and why each is on screen:
- `DEMO.md`: the walkthrough you'll reference; reads the four guard slots and fires three live blocks.
- `INSTRUCTIONS.md`: the four exercises; keep it open to point at each acceptance check.
- `INTERFACES.md`: the frozen signatures, including `generate` and `run_pipeline`. This is what "do not change the signature" means.
- `src/guardrails/input_guards.py`: the explainable regex fast-path; injection, PII, and system-prompt-leak patterns live here. Exercise 1 edits this.
- `src/guardrails/llm_guard/input_guards.py`: the DeBERTa and Presidio wrapper that catches the paraphrases regex can't.
- `src/guardrails/llm_judge/output_guards.py`: the hallucination judge you calibrate in Exercise 2.
- `src/guardrails/rate_limit.py`: the token-bucket limiter and output-token cap you wire in Exercise 3.
- `src/gateway/routes.py`: the live route where every guard is ordered into the chain.
- `src/generator.py`: where Exercise 3 threads `max_tokens`.
- `src/models.py`: the `QueryResponse` shape Exercise 4 constrains.
- `prompts/judge.j2`: the judge's rubric, one of the knobs you vary.

## 1. Intro  (~45-60s)

*[Stage: DEMO.md on screen.]*

In this module you'll harden a working question-and-answer app so it survives hostile input. The app is ScikitDocs, an assistant for the scikit-learn library. Its guardrails are already wired, so your job is to extend them and prove they fire.

Here's the framing to carry through. Guardrails reduce risk, they don't erase it. Every guard you add narrows one threat and leaves others open, and the docs are honest about that gap.

## 2. Topic overview  (~60-90s)

*[Stage: open src/gateway/routes.py and scroll the guard sequence.]*

A guardrail is a check that runs around the model, not inside it. The key idea is that these guards are layered and ordered. They form a chain, and the order is deliberate.

*[Stage: bring src/guardrails/input_guards.py forward.]*

Picture four seams. First, input checks on the raw question. Then injection detection, where someone tries to override your instructions. Then personally identifiable information, PII, handling, where an email gets redacted before anything downstream sees it. Last, the output side, where a judge reads the finished answer.

Each seam is its own concern, and you can extend one without touching the others. The misconception to flag: layering means composition, not substitution. The cheap regex layer short-circuits known attacks, and the slower model layer catches the rest. You keep both because they catch different things.

## 3. Exercise call-outs

### Exercise 1: Add a new input guard

*[Stage: switch to INSTRUCTIONS.md, Exercise 1; then point at src/guardrails/input_guards.py.]*

The first exercise adds one new guard to the live input stack. You'll pick from three options: an invisible-character check, a custom membership-number pattern, or a new system-prompt-leak phrasing.

The concept to hold first is where your guard plugs in. The input stack is ordered, so cheaper checks go earlier. An invisible-character scan is cheaper than the regex injection sweep, so it slots in ahead of it.

Here's the watch-out. None of these options is a complete defense, and the docs say so plainly. The value isn't comprehensiveness; it's wiring a guard into the chain and proving it fires. You're done when a test passes and a live request returns the reason string you expect.

### Exercise 2: Calibrate the LLM-judge

*[Stage: open src/guardrails/llm_judge/output_guards.py and prompts/judge.j2.]*

The second exercise calibrates the output judge. You'll build ten queries: five grounded answers and five with a wrong scikit-learn name, then score each with the judge.

Here's the crucial watch-out. The judge is probabilistic. It only flags an answer when the model actually hallucinates. So a grounded answer that correctly refuses will pass the judge, every time, on purpose. Don't expect the judge to fire on every row; a clean pass is a correct result, not a miss.

You're done with a one-page report holding two confusion matrices and a recommendation, honest about the small sample.

### Exercise 3: Wire the LLM10 consumption cap

*[Stage: open src/guardrails/rate_limit.py, then src/generator.py.]*

The third exercise caps unbounded consumption. You'll measure cost amplification, thread an output-token limit into the generator, then trigger the limiter under a burst.

Here's the watch-out that trips everyone. The rate limiter blocks on concurrent load, not on a slow sequential loop. Each request makes two model calls, so a sequential loop drains time and the sixty-second window rolls forward before it ever fills. Fire your requests in a tight burst to see the twenty-first one block.

You're done with a cost table, the patched generator, and a burst log.

### Exercise 4: Wire a Pydantic output validator

*[Stage: open src/models.py, then src/gateway/routes.py.]*

The fourth exercise enforces the response shape at the gateway boundary. You'll add constraints so an answer with no citations, or a confidence value out of range, fails loudly instead of slipping out silent.

The concept here is the contract. The service promises a shape, and you validate it on the way out. A broken contract is your fault, not the caller's, so the failure returns a five-oh-two, not a client error.

The watch-out is small but real: reset the rate-limit bucket at the top of each test, since the limiter runs before the validator and shared state leaks between tests. You're done when both tests pass and the markers are in place.

## 4. Key insights  (~30-45s)

*[Stage: return to src/gateway/routes.py and the guard sequence.]*

Three takeaways. First, defense is layered and ordered: each guard is its own seam, and you compose them rather than replace one with another. Second, a probabilistic judge passing a grounded answer is the system working, not failing. Third, a limiter that protects against bursts won't react to a slow loop, so test it the way an attacker would.

## 5. Outro  (~20-30s)

*[Stage: return to DEMO.md.]*

That's the guardrail stack: input checks, injection and PII seams, and an output judge with a consumption cap. The next module moves from blocking bad traffic to comparing good answers, where you'll run A/B tests on the pipeline. See you in the next one.
