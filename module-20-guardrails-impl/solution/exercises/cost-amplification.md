# Cost-amplification surface (Exercise 3, Step 1 reference solution)

Three passes through `traced_pipeline` with a fixed model (`gpt-4o-mini`). Produced
by `scripts/cost_amplification.py`; reproduce from `solution/` with
`PYTHONPATH=. uv run python scripts/cost_amplification.py`.

Pass 1 measures a baseline cohort. Pass 2 fires long-generation prompts uncapped and
records how long the answers actually run. Pass 3 derives an output cap from that
distribution (mean + 1 sd) and re-fires, so the cap only bites the tail.

> One real run. Token counts and costs drift a little across runs. Re-run and read
> your own; the method is the lesson.

## Cost per request

| cohort         | median  | p95     |
| -------------- | ------- | ------- |
| baseline       | $0.0004 | $0.0006 |
| long, uncapped | $0.0007 | $0.0010 |
| long, cap 857  | $0.0008 | $0.0009 |

Amplification (long p95 / baseline p95): uncapped **1.6x**, capped **1.5x**.

## Set the cap from the distribution, not a round number

The uncapped long answers ran **637 tokens on average, sd 220**. A round-number cap
is a guess. Set it from that distribution instead: at **mean + 1 sd = 857 tokens**
the cap sits above every normal answer and clips only the runaway tail.

Re-fired at 857, exactly **one of ten** rows hit the cap: the 925-token outlier was
clipped to 857, every other answer came in under it untouched. That single clip
pulled the p95 amplification from 1.6x down to 1.5x. The cap protected the normal
answers and bit only the one that ran away, which is the whole point. Lower the
multiplier (`--sd 0.5`) and it bites more of the tail, at the cost of clipping some
legitimate answers, so the multiplier is the knob you tune against your cost budget.

## Interpretation

The amplification surface here is small (about 1.6x), far below CVE-2025-53773's
up-to-20x on GitHub Copilot. The grounded generator answers from the retrieved
chunks and rarely runs past ~700 tokens, so there is little tail to cut. The same
grounding that stops hallucinations in Exercise 2 also bounds runaway length here. A
code-generation or open-ended-chat workload would show a much larger surface, and
the cap would earn its keep. The discipline is the same either way: measure your own
answer-length distribution, then place the cap above the normal range so it clips
abuse and leaves legitimate answers whole.
