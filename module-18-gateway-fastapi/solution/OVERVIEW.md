---
module_number: 18
module_title: "Build an LLM Gateway with FastAPI"
slug: gateway-fastapi
unit_type: overview-video-script
target_minutes: "5-7"
word_count: 719
---

# Module 18 Overview Video: Build an LLM Gateway with FastAPI

> Recording aid for a 5-7 min overview (read pace ~100 wpm). Read all prose aloud, top to bottom. `*[Stage: ...]*` lines are screen directions, not spoken. Not learner-facing.

## 0. VS Code setup (before you hit record)

Open the files you will show on camera (run from the repo root):

```bash
code exercises/18-gateway-fastapi/gateway-fastapi-starter/DEMO.md exercises/18-gateway-fastapi/gateway-fastapi-starter/INSTRUCTIONS.md exercises/18-gateway-fastapi/gateway-fastapi-starter/INTERFACES.md exercises/18-gateway-fastapi/gateway-fastapi-starter/src/config.py exercises/18-gateway-fastapi/gateway-fastapi-starter/src/gateway/classifier.py exercises/18-gateway-fastapi/gateway-fastapi-starter/prompts/classifier.j2 exercises/18-gateway-fastapi/gateway-fastapi-starter/src/gateway/router.py exercises/18-gateway-fastapi/gateway-fastapi-starter/src/generator.py exercises/18-gateway-fastapi/gateway-fastapi-starter/src/gateway/routes.py exercises/18-gateway-fastapi/gateway-fastapi-starter/src/gateway/providers/anthropic.py exercises/18-gateway-fastapi/gateway-fastapi-starter/src/pricing.py
```

Files and why each is on screen:
- `DEMO.md`: the walkthrough you will reference; reads the gateway top to bottom and fires a request through it.
- `INSTRUCTIONS.md`: the three exercises; keep it open to point at each acceptance criterion.
- `INTERFACES.md`: the frozen signatures the adapter and pipeline must match. This is what "do not change the signature" means.
- `src/config.py`: the `Settings` object; Exercise 1 adds the `model_premium` name here.
- `src/gateway/classifier.py`: the self-classifying tier picker; the `QueryType` literal you extend lives here.
- `prompts/classifier.j2`: the prompt that teaches the classifier its labels; the load-bearing output-format line is here.
- `src/gateway/router.py`: the ten-line dispatch; `select_model` and `route_query` are what the exercises edit.
- `src/generator.py`: the chat-completions call site Exercise 2 wraps with retry.
- `src/gateway/routes.py`: the `/query` handler and the `QueryRequest` model that gains a `provider` field.
- `src/gateway/providers/anthropic.py`: the stub you replace with the Anthropic adapter in Exercise 3.
- `src/pricing.py`: the per-model rate table; you add one synthetic Anthropic row here.

## 1. Intro  (~45-60s)

*[Stage: DEMO.md on screen.]*

In this module you build a gateway: one front door that every large language model request flows through. Large language model is shortened to LLM. By the end you'll have added a third routing tier, wrapped the model call in retries, and bolted on a second provider.

Here's why it matters. Without a gateway, every service reinvents secrets, tracing, and routing. Put those in one place and you solve each problem once. You won't write much code. The skill is choosing what the gateway should own.

## 2. Topic overview  (~60-90s)

*[Stage: open src/gateway/router.py, point to the route_query function.]*

A gateway hides the messy differences between providers behind one stable response shape. Ask a question, and you get back the same fields every time: an answer, sources, a cost, a model name. What happens behind that shape can change without the caller noticing.

Routing is the first job. The gateway reads your question, decides if it's simple, complex, or premium, and picks the model to match. Cheap model for the easy stuff, stronger model for the hard.

Here's the mental model to hold onto. The gateway is a thin seam, not a thick layer. Each capability is a few lines, and they compose into one short function. And here's the misconception that trips people up: a gateway isn't an extra server to babysit. It's the one place your cross-cutting concerns live, so the services behind it stay simple.

## 3. Exercise call-outs

### Exercise 1: Extend the classifier to a third tier

*[Stage: switch to INSTRUCTIONS.md, Exercise 1; then point at prompts/classifier.j2.]*

The first exercise adds a premium tier for long-context or high-stakes queries. It's a four-step edit: a new model name in settings, a new label in the classifier, a new rubric line in the prompt, and a new branch in the dispatch.

Here's what to watch out for. The classifier is a small model reading your query and labeling it. So the prompt does the real work. If you teach the rubric the new label but leave the output-format line saying only two values are allowed, the model is told premium is off-limits and almost never picks it. Edit every place the prompt counts the tiers.

You're done when a five-query run shows a mix of tiers. Since premium shares a model with complex for now, read the tier from the cost log's `query_type` field, not the model name.

### Exercise 2: Wrap the OpenAI client call with tenacity retries

*[Stage: open src/generator.py, point to the chat-completions call.]*

The second exercise adds retries with the tenacity library. The pattern has three parts: exponential backoff so retries space out, jitter so many clients don't all retry at the same instant, and a tight filter on which errors retry.

Here's the watch-out, and it's the whole lesson. Retry the transient server errors, the five-hundred family, because those usually clear on a second try. But fail fast on client errors, the four-hundred family. A four-hundred means your request was wrong, so retrying just re-sends the same wrong request. Catch-everything retry is what teams ship by accident, and a retry storm teaches them to stop.

You're done when two tests pass: one where two server errors resolve on the third try, one where a four-hundred fails immediately with no retry.

### Exercise 3: Add a thin Anthropic adapter for multi-provider routing

*[Stage: open src/gateway/providers/anthropic.py, then INTERFACES.md to show the frozen return shape.]*

The third exercise wires a second provider behind the same response shape. You write a thin adapter that converts the gateway's input to Anthropic's Messages format and converts the reply back. The Anthropic call is a stub returning a canned answer, so you don't need an Anthropic key yet. The pattern is the deliverable, not the credentials.

Here's what to watch out for. The semantic cache key is the question text alone, and it's provider-agnostic. So if you ask the same question through OpenAI, then through Anthropic, the second call just returns the first one's cached answer. Clear the cache between the two calls or the comparison is meaningless.

You're done when two side-by-side requests return the identical shape, with the model field reflecting each provider's choice.

## 4. Key insights  (~30-45s)

Three takeaways. First, the gateway's value is the stable response shape: providers and tiers change underneath while callers never flinch. Second, a retry policy is defined by what it refuses to retry, so naming your no-retry conditions matters more than the backoff math. Third, multi-provider support is just conversion at the adapter boundary: pick one canonical shape and remap everything else into it.

## 5. Outro  (~20-30s)

*[Stage: return to DEMO.md.]*

That's the gateway: one door, three jobs, one response shape. Every later layer bolts on here, because every request already flows through. Guardrails and A/B testing plug into this same seam. See you in the next one.
