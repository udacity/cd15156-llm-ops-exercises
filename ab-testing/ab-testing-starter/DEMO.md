> A walkthrough of the codebase you'll work with. See INSTRUCTIONS.md for the exercise tasks.

# Demo — Build a Sticky-by-User Feature Flag, Route Two Prompt Variants, Log Per-Call

The previous concept module named four LLM-specific A/B pitfalls — cost asymmetry, sticky-versus-per-request assignment, judge variance, and distribution drift — and three vendor axes for feature-flagging products. This demo brings two of those pitfalls down to code in the starter. You will read three primitives the starter ships under `src/optimization/ab.py`: a deterministic SHA-256 hash-mod `pick_variant(client_id, traffic_split, salt)` that buckets the same user onto the same variant every call, a `call_with_variant` that renders one of two Jinja prompt templates and calls OpenAI through the same Vocareum-or-direct bridge `src/generator.py` uses, and a `log_assignment` helper that writes one JSONL row per call for Exercise 2's chi-squared analyzer. Production teams use OpenFeature-backed flags through LaunchDarkly, Statsig, or Flipt — the in-process flag here is good enough for this exercise, and the seams it leaves match what production systems expose.

The starter is structured so the gateway already carries the contract this demo consumes. The gateway module added an `X-Client-Id` header to `POST /query` that threads through `src/gateway/routes.py:48-62` into `src/gateway/router.py:38-76` as the `client_id` keyword. Demo Part 3 shows how that header is exactly what `pick_variant` needs to hash. The capstone (a separate codebase outside this starter) is frozen and uses per-request weighted sampling because its workload has no user identifier — both choices appear here, both are named as production tradeoffs.

## Part 1 — Sticky-by-user `pick_variant` via SHA-256

Open `src/optimization/ab.py` and read `pick_variant` at lines 53-113. The shape is small:

```python
def pick_variant(client_id, traffic_split, *, salt=""):
    # ... validation ...
    if not client_id:
        return random.choices(variants, weights=weights, k=1)[0]
    digest = hashlib.sha256((salt + client_id).encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 1_000_000
    threshold = bucket / 1_000_000 * total
    running = 0.0
    for variant, weight in zip(variants, weights):
        running += weight
        if threshold < running:
            return variant
    return variants[-1]
```

Three things to notice. The hash is content-addressable — there is no process-local state, no shared store, no database lookup. The same `client_id` produces the same bucket whether the call lands on the gateway in San Francisco or in Frankfurt, today or six months from now. That property is what "sticky" means in production. LaunchDarkly's percentage-rollout documentation calls it deterministic bucketing; Statsig's experiment configuration docs use the same primitive under a different label; the OpenFeature specification leaves the hashing scheme to the provider but expects the same `(flag, evaluation_context)` pair to return the same result. SHA-256 mod is the standard production choice because it distributes uniformly without correlating with any structural property of the identifier — sequential `user-1`, `user-2`, `user-3` ids land on independent buckets.

The `salt` parameter is the production discipline. Two concurrent A/B experiments that hash the same `client_id` with no salt would produce correlated assignments — a user who saw variant B in experiment 1 would always see variant B in experiment 2, and a metric shift you attribute to experiment 2 might really come from leftover effects of experiment 1. Salting with a per-experiment string — LaunchDarkly calls it "salt," Statsig calls it "experiment key" — fixes it. The `tests/test_ab.py::test_pick_variant_salt_isolates_experiments` test exercises this directly: over 500 client_ids, swapping the salt makes roughly half of them flip variants, which is exactly what independence predicts.

Why SHA-256 specifically. A non-cryptographic hash like MD5 or CityHash would also distribute uniformly, and either would be roughly an order of magnitude faster — perfectly acceptable for a flag function called once per request. The reason to reach for SHA-256 anyway is that Python's standard library ships it without an extra dependency (no `pip install xxhash`), the per-call cost is microseconds even for a busy gateway, and the property that matters for A/B routing is uniformity of the digest's leading bits, which all three give you. The cumulative-weight bucket loop at the bottom is the standard inverse-CDF construction: walk the variants in order, accumulate the weight, return the first variant whose running total exceeds the threshold. The trailing `return variants[-1]` line handles the floating-point edge where `threshold` exactly equals the accumulated total — without it, a degenerate `{"A": 0.0, "B": 1.0}` split could theoretically fall off the end.

The `client_id=None` fallback is honest, not lazy. The capstone calls `pick_variant(None, ...)` because its FAQ workload has no user identifier — there is no auth, no session cookie, no API-key scoping. Falling back to weighted random sampling lets the same function serve both workloads without raising. The concept module named the sticky-versus-per-request tradeoff explicitly; the impl side of that conversation is one function that does both, with the choice of which path you're on visible in whether you pass a `client_id` value.

## Part 2 — `call_with_variant` and the two templates

The router is `call_with_variant` at `src/optimization/ab.py:116-159`. It picks the template name from the variant key, renders it with the retrieved chunks using the same Jinja `Environment(loader=FileSystemLoader(...), keep_trailing_newline=True, autoescape=False)` idiom from `src/generator.py:34-38`, and calls OpenAI through the Vocareum-or-direct bridge:

```python
template = _env.get_template(f"docbot_system_{variant}.j2")
contexts = "\n\n---\n\n".join(s.chunk_text for s in sources)
system_prompt = template.render(contexts=contexts)
client = OpenAI(base_url=settings.openai_base_url or None)
t0 = time.perf_counter()
response = client.chat.completions.create(
    model=chosen_model,
    temperature=constants.GENERATION_TEMPERATURE,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ],
)
latency_ms = int((time.perf_counter() - t0) * 1000)
