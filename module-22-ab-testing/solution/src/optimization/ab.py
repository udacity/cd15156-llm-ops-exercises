"""Sticky-by-user A/B routing + variant-aware OpenAI caller.

Three primitives, intentionally small:

1. :func:`pick_variant` — deterministic SHA-256 hash-mod bucketing over
   ``(salt, client_id)``. Same ``client_id`` + same ``salt`` always
   returns the same variant. ``client_id=None`` falls back to weighted
   random sampling so the function is safe to call on un-headered
   traffic.
2. :func:`call_with_variant` — renders ``prompts/docbot_system_{V}.j2``
   with the retrieved chunks, calls OpenAI through the Vocareum-or-
   direct bridge, returns ``(answer, TokenUsage, cost_usd, latency_ms)``.
3. :func:`log_assignment` — appends one JSONL row per call to a path
   the analyzer (``scripts/ab_analyze.py``) can read.

Two design choices worth naming:

- **Salt isolates experiments.** Two concurrent experiments that hash
  the same user with no salt would give correlated assignments — a
  user who saw variant B in experiment 1 would always see variant B in
  experiment 2. Salting by experiment key (LaunchDarkly calls it
  "salt"; Statsig calls it "experiment ID") is the standard fix.
- **Per-request fallback is honest, not lazy.** A workload with no
  user-identifier surface can call ``pick_variant(client_id=None, ...)``
  and get per-request weighted sampling. That's the right behavior for
  that workload — naming it as a deliberate fallback rather than a
  missing feature is the discipline.

The ``X-Client-Id`` request header arrives at the gateway's
``route_query`` as the ``client_id`` keyword; the caller composition
reads that value and feeds it to ``pick_variant``.
"""

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from openai import OpenAI

from src import constants
from src.config import settings
from src.models import Source, TokenUsage
from src.pricing import compute_cost

# Same prompts directory as ``src.generator`` — variants live next to the
# base ``docbot_system.j2`` so the loader picks them up without config.
_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
_env = Environment(
    loader=FileSystemLoader(_PROMPTS_DIR),
    keep_trailing_newline=True,
    autoescape=False,
)


def pick_variant(
    client_id: str | None,
    traffic_split: dict[str, float],
    *,
    salt: str = "",
) -> str:
    """Return the variant assigned to ``client_id`` under ``traffic_split``.

    When ``client_id`` is non-empty the assignment is sticky: the same
    ``client_id`` + same ``salt`` + same ``traffic_split`` always returns
    the same variant. The hash is SHA-256 over ``salt + client_id``
    interpreted as a 64-bit integer, mod the cumulative-weight bucket.

    When ``client_id`` is ``None`` or empty the function falls back to
    weighted random sampling via :func:`random.choices`. That keeps the
    contract safe to call on un-headered traffic (an evaluation harness,
    a workload with no user identifier) without raising.

    Args:
        client_id: Bucketing key. Typically the value of the
            ``X-Client-Id`` request header (per
            ``constants.CLIENT_ID_HEADER``). ``None`` triggers the
            per-request fallback.
        traffic_split: Mapping of variant name to non-negative weight.
            Weights need not sum to 1 — they're normalised by their
            sum. ``{"A": 10, "B": 30}`` is equivalent to
            ``{"A": 0.25, "B": 0.75}``.
        salt: Optional experiment identifier. Two concurrent experiments
            should use different salts so the same ``client_id`` lands
            on independent variant choices.

    Returns:
        One of the keys of ``traffic_split``.

    Raises:
        ValueError: If ``traffic_split`` is empty, has a negative
            weight, or has weights summing to zero.
    """
    if not traffic_split:
        raise ValueError("traffic_split must contain at least one variant")
    if any(w < 0 for w in traffic_split.values()):
        raise ValueError("traffic_split weights must be non-negative")
    total = sum(traffic_split.values())
    if total <= 0:
        raise ValueError("traffic_split weights must sum to a positive value")

    variants = list(traffic_split.keys())
    weights = [traffic_split[v] for v in variants]

    if not client_id:
        # Per-request fallback. ``random.choices`` normalises weights
        # internally so we pass them as-is.
        return random.choices(variants, weights=weights, k=1)[0]

    # SHA-256 of (salt + client_id) → first 8 bytes → 64-bit int → mod
    # the cumulative-weight bucket. Standard production primitive; same
    # shape LaunchDarkly and Statsig use under the hood.
    digest = hashlib.sha256((salt + client_id).encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 1_000_000
    threshold = bucket / 1_000_000 * total

    running = 0.0
    for variant, weight in zip(variants, weights):
        running += weight
        if threshold < running:
            return variant
    # Floating-point edge: ``threshold`` can equal ``total`` after the
    # accumulation. Returning the last variant is the right behavior.
    return variants[-1]


def call_with_variant(
    question: str,
    sources: list[Source],
    variant: str,
    *,
    model: str | None = None,
) -> tuple[str, TokenUsage, float, int]:
    """Render ``docbot_system_{variant}.j2`` and call OpenAI.

    Same control flow as :func:`src.generator.generate` with two
    differences: (1) the template name varies per variant, and
    (2) the return tuple adds ``latency_ms`` so the per-variant latency
    aggregate in Exercise 3 has a value to read.

    Args:
        question: The user's question.
        sources: Retrieved chunks (as produced by ``src.store.query``).
        variant: Variant key — must match a file at
            ``prompts/docbot_system_{variant}.j2``.
        model: Model name override. Defaults to
            ``settings.model_simple`` (``gpt-4o-mini``) so the cost-
            asymmetry warning in the exercise is anchored on the
            cheaper tier.

    Returns:
        ``(answer, TokenUsage, cost_usd, latency_ms)``.
    """
    chosen_model = model or settings.model_simple
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
    answer = response.choices[0].message.content or ""
    usage = TokenUsage(
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
    )
    cost_usd = compute_cost(chosen_model, usage)
    return answer, usage, cost_usd, latency_ms


def log_assignment(
    path: Path | str,
    *,
    client_id: str | None,
    variant: str,
    question: str,
    answer: str,
    usage: TokenUsage,
    cost_usd: float,
    latency_ms: int,
    success: bool,
) -> None:
    """Append one JSONL row capturing a single A/B call.

    The schema is the one ``scripts/ab_analyze.py`` reads. Keep the
    field set stable — adding or renaming fields breaks the analyzer.

    Args:
        path: Destination file. Parent directories are created if
            missing so callers don't have to ``mkdir`` themselves.
        client_id: The bucketing key. ``None`` is written as JSON null
            so the analyzer can distinguish sticky from per-request
            rows after the fact.
        variant: The variant the row belongs to.
        question: The user's question (truncated to the first 500 chars
            so a runaway question doesn't bloat the log file).
        answer: The model's response (truncated to 2000 chars for the
            same reason).
        usage: Token usage from the OpenAI response.
        cost_usd: Per-call cost from ``compute_cost``.
        latency_ms: Wall-clock latency of the OpenAI call in
            milliseconds.
        success: The boolean success heuristic the exercise defines.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "client_id": client_id,
        "variant": variant,
        "question": question[:500],
        "answer": answer[:2000],
        "latency_ms": latency_ms,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "cost_usd": cost_usd,
        "success": success,
    }
    with destination.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
