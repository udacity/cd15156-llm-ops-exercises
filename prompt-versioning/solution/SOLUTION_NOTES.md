# Solution Notes — Module 03 Prompt Versioning

These notes cover the three exercises from `INSTRUCTIONS.md`. Files
added or modified relative to the starter:

## Exercise 1 — Template extension

- `prompts/docbot_system.j2` — added instruction 8 wrapped in
  `{% if user_tier == "premium" %}` ... `{% endif %}`.
- `src/generator.py` — `render_system_prompt` now accepts
  `user_tier: str = "standard"` and threads it into `template.render(...)`.
  Default value preserves backward compatibility with existing callers.
- `tests/test_prompt_tier.py` — two tests asserting premium output
  contains the mailing-list pointer and standard output does not.

## Exercise 2 — Env-aware prompt loader

- `prompts/prod/docbot_system.j2` — clean copy of the post-Exercise-1
  template (unchanged behavior).
- `prompts/dev/docbot_system.j2` — same template plus a `## Debug mode`
  suffix asking the model to wrap answers in `[DEV]` / `[/DEV]` markers.
- `src/prompts/__init__.py` — empty marker, makes `src.prompts` a
  package.
- `src/prompts/loader.py` — `load_environment(env_name=None)` reads
  `PROMPT_ENV` (defaulting to `"prod"`), validates the value, and
  returns a Jinja `Environment` rooted at `prompts/<env>/`.
- `src/config.py` — added `prompt_env: Literal["dev", "prod"] = "prod"`
  on `Settings` per the Exercise-2 hint. The loader still reads
  `os.environ` directly so tests using `monkeypatch.setenv` work without
  re-instantiating the `Settings` singleton; learners who want a
  pydantic-driven loader can swap the body for
  `env_name = env_name or settings.prompt_env`.
- `tests/test_prompt_loader.py` — four tests covering the default-prod
  path, the dev-template-loads-dev path, the explicit-override path,
  and the invalid-value rejection path.

## Exercise 3 — A/B refusal-rate harness

- `scripts/ab_refusal.py` — runs ten fixed questions (five answerable,
  five out-of-scope) against `main` and `ex3-soft-refusal` branches by
  `git checkout`-ing between variants. Classifies each answer with a
  regex, prints a chi-squared p-value, and reports `significant_at_0.05`
  as a JSON object on stdout.
- The branch and prompt-edit step (changing instruction 3 to "Lean
  toward answering...") is a learner workflow step, not a file change
  shipped here — the harness assumes the learner has made the commit
  on `ex3-soft-refusal`. Without that branch, `subprocess.run(["git",
  "checkout", "ex3-soft-refusal"], check=True)` will raise.

### Exercise 3 — Reflection answers (non-code deliverable)

The exercise asks two written questions that don't map to code. The
expected answer shape is below; learners self-grade against this:

1. **Sample size for a 20-percentage-point detectable effect at 95%
   confidence.** Order of magnitude: a few hundred per variant. A rough
   rule of thumb for a two-proportion z-test at p1 = 0.50, p2 = 0.30,
   alpha = 0.05, power = 0.80 puts each group at roughly 90-100
   samples; the concept module's 10k-20k requests-per-variant figure
   was for a 5% relative shift in a binary metric, which is a much
   smaller effect and thus a much larger required sample. The point of
   the question is that 10 per variant is two orders of magnitude
   below detection capacity, so a "no significant difference" reading
   here is uninformative either way.

2. **Confirming the refusal-rate shift is real (not a labeling
   artifact).** Acceptable answers include: (a) replace the regex with
   an LLM-as-judge that scores each answer as refused / answered /
   partially-answered with a rubric; (b) human-label a holdout sample
   and measure regex agreement vs the human labels; (c) measure a
   downstream metric that does not depend on refusal labeling — e.g.,
   thumbs-up rate, follow-up-question rate, or task-completion rate
   in a UX session. Any one of these moves the experiment off the
   regex's reliability floor.

## KNOWN-LIMITATIONS

- `scripts/ab_refusal.py` imports `requests` and `scipy.stats`, neither
  of which is in the starter's `pyproject.toml`. The exercise text in
  `INSTRUCTIONS.md` reproduces the script verbatim, so the script is
  shipped as written. Learners need to add the deps before running it:
  `uv add requests scipy`. (The starter already pulls in `httpx`, which
  could substitute for `requests` with a small edit, but we keep the
  exercise's code unchanged so learners see exactly what the
  instructions described.) Central verification will install whatever
  the centralised env spec defines.
- The reflection text in Exercise 3 is non-code; see "Reflection
  answers" above.
