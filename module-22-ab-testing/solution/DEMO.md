> A walkthrough of the codebase you'll work with. See INSTRUCTIONS.md for the exercise tasks.

# Demo — Build a Sticky-by-User Feature Flag, Route Two Prompt Variants, Log Per-Call

There are four LLM-specific A/B pitfalls — cost asymmetry, sticky-versus-per-request assignment, judge variance, and distribution drift — plus three vendor axes for feature-flagging products. This demo brings two of those pitfalls down to code in the starter. You will read three primitives the starter ships under `src/optimization/ab.py`: a deterministic SHA-256 hash-mod `pick_variant(client_id, traffic_split, salt)` that buckets the same user onto the same variant every call, a `call_with_variant` that renders one of two Jinja prompt templates and calls OpenAI through the same Vocareum-or-direct bridge `src/generator.py` uses, and a `log_assignment` helper that writes one JSONL row per call for Exercise 2's chi-squared analyzer. Production teams use OpenFeature-backed flags through LaunchDarkly, Statsig, or Flipt — the in-process flag here is good enough for this exercise, and the seams it leaves match what production systems expose.

The starter is structured so the gateway already carries the contract this demo consumes. The gateway provides an `X-Client-Id` header on `POST /query` that threads through `src/gateway/routes.py:48-62` into `src/gateway/router.py:38-76` as the `client_id` keyword — that header is exactly what `pick_variant` needs to hash. A workload with no user identifier can instead use per-request weighted sampling; both choices appear here, both are named as production tradeoffs.

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

The `client_id=None` fallback is honest, not lazy. A workload with no user identifier — no auth, no session cookie, no API-key scoping — calls `pick_variant(None, ...)`. Falling back to weighted random sampling lets the same function serve both kinds of workload without raising. The sticky-versus-per-request tradeoff comes down to one function that does both, with the choice of which path you're on visible in whether you pass a `client_id` value.

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
# ... extract answer, usage, compute cost ...
return answer, usage, cost_usd, latency_ms
```

Two design choices worth naming. The `base_url=settings.openai_base_url or None` line is the same bridge `src.generator` uses: when `OPENAI_BASE_URL` is empty (direct OpenAI), the SDK uses its default; when it is set to the Vocareum URL, the SDK routes through the proxy. Same code, two deploy targets, no conditional. The exercise inherits this without any new configuration. The return tuple adds `latency_ms` next to `(answer, TokenUsage, cost_usd)` — `src.generator.generate` returns only the first three because the cost-tracking path did not need wall-clock latency, but A/B analysis does. Exercise 3 reads `latency_ms` for the per-variant latency aggregate.

The two templates live next to the base prompt at `prompts/`. `docbot_system_A.j2` is a verbatim copy of `docbot_system.j2`. `docbot_system_B.j2` differs in exactly one instruction — number 4 changes from "Be concise and direct" to "Be expansive" with a request to add a sentence or two of related context from the documentation. Multi-variable variant drift is a source of "the two variants are different in seven ways and you cannot attribute the metric shift to any one of them"; constraining the diff to one instruction is the experimental-design discipline. The `tests/test_ab.py::test_call_with_variant_renders_correct_template` test pins this by checking that "Be expansive" appears in variant B's system prompt and not in variant A's.

`log_assignment` at `src/optimization/ab.py:162-211` writes one JSONL row per call to a path the analyzer reads. The schema is small and stable: `client_id`, `variant`, `question` (truncated to 500 chars), `answer` (truncated to 2000), `latency_ms`, `prompt_tokens`, `completion_tokens`, `cost_usd`, and `success`. The `client_id` field is preserved as null when the per-request fallback ran, so the analyzer can distinguish sticky rows from fallback rows after the fact — useful when you want to compare the two assignment schemes on the same dataset.

## Part 3 — Wiring through the gateway and the honest seam

The gateway provides the `X-Client-Id` header on its `POST /query` route. Read `src/gateway/routes.py:48-62`:

```python
@router.post(constants.QUERY_ROUTE, response_model=QueryResponse)
def query_endpoint(
    request: QueryRequest,
    client_id: Annotated[
        str | None, Header(alias=constants.CLIENT_ID_HEADER)
    ] = None,
) -> QueryResponse:
    return route_query(
        request.question,
        top_k=request.top_k,
        client_id=client_id,
    )
```

The header value flows to `route_query(..., client_id=client_id)` at `src/gateway/router.py:38-76`. Today `route_query` accepts the keyword and forwards it for downstream use without consuming it — that contract leaves a clean place to hook in A/B routing. To turn on A/B for the gateway, you wrap `route_query`:

```python
def ab_route_query(question, *, client_id, traffic_split, salt):
    variant = pick_variant(client_id, traffic_split, salt=salt)
    # ... retrieve sources via the same pipeline route_query uses ...
    return call_with_variant(question, sources, variant)
```

The honest seam: this walkthrough does not modify `src/gateway/router.py` to permanently swap `route_query` for `ab_route_query`. Exercise 1 wraps it from a script instead, which keeps the cache, tracing, cost-logging, and classifier layers `route_query` already composes from being disturbed. In a production retrofit you would push the variant selection into `route_query` itself, log the variant onto the trace span, and let the existing observability flow carry the A/B labels downstream. That refactor is on the order of a day's work and out of scope for a thirty-minute module; the exercise's harness is the right shape for a development-time experiment.

Run the demo once from the project root with a tiny `client_id` pool to confirm the wiring:

```python
from src.models import Source
from src.optimization import pick_variant, call_with_variant
sources = [Source(doc_id="d1", chunk_text="DBSCAN clusters by density.", similarity_score=0.95)]
for cid in ["alice", "bob", "carol", "alice"]:  # alice appears twice
    v = pick_variant(cid, {"A": 0.5, "B": 0.5}, salt="prompt-style-v1")
    print(cid, v)
```

Four lines, alice's two assignments identical, bob and carol independent. That is sticky-by-user. To close, name what this walkthrough addresses from the four LLM-specific pitfalls and what it does not. Cost asymmetry is handled — `log_assignment` carries `cost_usd` per call, so Exercise 3's analyzer can break it down by variant and surface a price-per-call difference that a chi-squared on success rate alone would miss. Sticky-by-user assignment is the deliberate design — the gateway header, the hash function, and the salt all exist for it. Per-request assignment is the documented fallback for a user-less workload. Judge variance is handled head-on — the success metric in Exercise 1 is itself an LLM-as-judge faithfulness call, so the single-call-versus-averaged-calls tradeoff is live and Exercise 3's stretch quantifies it; the same calibration discipline keeps the label trustworthy. Distribution drift would need wall-clock time the simulation does not consume. Three pitfalls handled, one named-and-fallback, one named-and-deferred. Bring the same discipline forward when you ship a real A/B harness behind the gateway.
