---
module_number: 11
module_title: "Build an Automated Evaluation Suite with RAGAS"
slug: evaluation-ragas
unit_type: overview-video-script
target_minutes: "5-7"
word_count: 713
---

# Module 11 Overview Video: Evaluation with RAGAS

> Recording aid for a 5-7 min overview (read pace ~100 wpm). Read all prose aloud, top to bottom. `*[Stage: ...]*` lines are screen directions, not spoken. Not learner-facing.

## 0. VS Code setup (before you hit record)

Open the files you will show on camera (run from the repo root):

```bash
code exercises/11-evaluation-ragas/evaluation-ragas-starter/DEMO.md exercises/11-evaluation-ragas/evaluation-ragas-starter/INSTRUCTIONS.md exercises/11-evaluation-ragas/evaluation-ragas-starter/INTERFACES.md exercises/11-evaluation-ragas/evaluation-ragas-starter/data/golden_set.csv exercises/11-evaluation-ragas/evaluation-ragas-starter/scripts/run_eval.py exercises/11-evaluation-ragas/evaluation-ragas-starter/scripts/eval_topk_sweep.py exercises/11-evaluation-ragas/evaluation-ragas-starter/src/evaluation/run_eval.py exercises/11-evaluation-ragas/evaluation-ragas-starter/src/evaluation/deprecated_apis.py
```

Files and why each is on screen:
- `DEMO.md`: the walkthrough you will reference; shows `make eval` running and how to read the five numbers.
- `INSTRUCTIONS.md`: the four exercises; keep it open to point at each acceptance criterion.
- `INTERFACES.md`: the frozen signatures, including `run_pipeline`, which the eval harness calls once per golden row.
- `data/golden_set.csv`: the thirty-row ground truth; Exercise 1 adds five rows here.
- `scripts/run_eval.py`: the eval entry point; Exercise 4 adds the two threshold flags here.
- `scripts/eval_topk_sweep.py`: the sweep harness Exercise 2 runs at top-k 3, 5, and 10.
- `src/evaluation/run_eval.py`: where the four-metric stack and the judge wrapper are declared.
- `src/evaluation/deprecated_apis.py`: the eight-entry allow-list behind the deprecated-API sub-metric.

## 1. Intro  (~45-60s)

*[Stage: DEMO.md on screen.]*

In this module you build the part of a retrieval-augmented generation app that tells you whether it's actually any good: the evaluation suite. You'll use a tool called retrieval-augmented generation assessment, or RAGAS, to score answers automatically.

So far you've built a pipeline that retrieves chunks and writes answers. But how do you know the answers are right? By the end you'll have extended the test set, swept a retrieval setting, diagnosed two failures, and wired a gate that fails a build when quality drops.

## 2. Topic overview  (~60-90s)

*[Stage: open src/evaluation/run_eval.py and point to the four-metric list.]*

RAGAS scores a RAG answer on four things. Faithfulness asks: are the claims actually supported by the retrieved text? Answer relevancy asks: is the answer on-topic? Context recall asks: did retrieval pull in the text it needed? Context precision asks: how much noise came along for the ride?

Here's the one idea to hold onto. These four measure different things, and they don't move together. A high context recall can sit right next to a low faithfulness. That means retrieval found the right text, but the generator ignored it and made something up. So you score them separately, and you read them together.

*[Stage: point to src/evaluation/deprecated_apis.py.]*

This starter adds a fifth, library-specific check: did the answer recommend a scikit-learn function that's been removed? RAGAS won't catch that on its own, so it gets its own score.

## 3. Exercise call-outs

### Exercise 1: Extend the golden set with five new rows

*[Stage: switch to INSTRUCTIONS.md, Exercise 1; then show data/golden_set.csv.]*

The golden set is your answer key: thirty questions with expected answers. The first exercise adds five of your own. The concept to grasp first is variance. You want a deliberate mix: a couple of easy lookups, a couple of compositional questions, and one hard one.

Here's what to watch out for. If every row you write scores above zero point nine, you've learned nothing. The whole point is spread. You want at least one row that drops below zero point six on some metric. That low score is the signal you'll diagnose later. You're done when the scores show a real spread, not a flat wall of nines.

### Exercise 2: Sweep top-k at 3, 5, and 10

*[Stage: open scripts/eval_topk_sweep.py, then point to the sample table in INSTRUCTIONS.md.]*

Top-k is how many chunks you retrieve per question. This exercise runs the eval at three, five, and ten, then picks one. The concept is a tradeoff: more chunks raise recall but lower precision, because noise rides along.

Now the big watch-out, and it's about your time. This sweep is slow. It fires the full judge stack three times over, so plan for tens of minutes, not a couple, and watch your interface quota. Start it and step away. You're done with a small table and one sentence recommending a value, usually five.

### Exercise 3: Diagnose two failures

*[Stage: switch to INSTRUCTIONS.md, Exercise 3.]*

This is the exercise that matters most. You pick your two lowest-scoring rows and name why each failed. Low recall with low precision means retrieval failed. High recall but low faithfulness means the generator failed. Off-topic everything-else-fine means routing failed.

Here's the watch-out that protects you from chasing ghosts. These scores are judged by a language model, so they jitter run to run. Read the shape, not the third decimal. With only thirty rows, one low score is a hint, not a verdict. You're done when each fix you propose is concrete and traces straight back to a number.

### Exercise 4: Configure a threshold gate for CI

*[Stage: open scripts/run_eval.py.]*

The last exercise turns the suite into an alarm. You add two flags that set a floor on faithfulness and on recall. If the scores drop below the floor, the script exits with an error code so a continuous-integration build fails.

The thing to watch out for is the exit code. Use exit code two, not one, so it doesn't collide with a normal build error. You're done when you've saved two transcripts: one healthy run that passes, and one degraded run that names the failed metric and fails.

## 4. Key insights  (~30-45s)

*[Stage: return to the four-metric list in src/evaluation/run_eval.py.]*

Three takeaways. First, the metrics measure different failures, so a perfect recall next to a poor faithfulness isn't a contradiction, it's a diagnosis. Second, these are model-judged numbers that wobble, so treat a five-point gap as noise and a twenty-point gap as signal. Third, evaluation only earns its keep when it gates a build, which is why the suite ends as a pass-or-fail check, not a pretty report.

## 5. Outro  (~20-30s)

*[Stage: return to DEMO.md.]*

That's the evaluation suite: four RAGAS scores, one library check, and a gate that holds the line. Later modules build on it, where guardrails catch the faithfulness drops you diagnosed here. See you in the next one.
