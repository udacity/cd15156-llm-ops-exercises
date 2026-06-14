# Solution notes — Module 24 (RAGOps Pipeline)

The starter already ships every source file this module exercises: `src/ingestion/watcher.py`, `src/ingestion/alias.py`, `src/ingestion/migrate.py`, `src/store.py` (alias-resolving `get_collection`), `scripts/start_watcher.py`, `scripts/migrate_blue_green.py`, plus the inbox templates at `data/docs_inbox-templates/{good,bad_invalid_json,bad_missing_field}.json` and the `make watch` / `make migrate-blue-green` Makefile targets. The exercises are operational — drop files, run scripts, read logs, query the gateway — and the deliverables are therefore *expected outputs* rather than authored code.

## Files these notes add

None. This directory mirrors the starter exactly. The only difference is this `SOLUTION_NOTES.md`, which captures the expected outputs and writeup-style deliverables for each exercise.

If you were grading, the artifacts to look for are: a populated `data/ACTIVE_COLLECTION` file naming a color (`scikit_docs_green` after Part B of Exercise 3), a non-empty `data/docs_inbox/failed/` directory with two `.error.txt` siblings after Exercise 2, and a successful `/query` response that cites the `modules.inbox_demo.intro` chunk after Exercise 1.

## Exercise 1 — Expected outputs

After `cp data/docs_inbox-templates/good.json data/docs_inbox/my_first_section.json` the watcher log should print (within ~1 second):

```
ingestion.watcher: Ingested my_first_section.json → modules.inbox_demo.intro#<12-char-hash> (active=scikit_docs)
```

The `/query` call should return a JSON body with:

- `answer` mentioning "inbox watcher" and the content-hashed chunk id design
- `sources` array including an entry whose id starts with `modules.inbox_demo.intro#`
- The new section in the top 5 hits by score

The dropped file remains in `data/docs_inbox/my_first_section.json` (the watcher does not delete on success — the inbox is the audit trail).

### Exercise 1 stretch — idempotency proof

After two drops of the same `good.json` payload under different filenames, the Chroma query should report exactly:

```
row count: 1
chunk id: modules.inbox_demo.intro#<deterministic-12-char-hash>
```

The deterministic hash is `sha256(text)[:12]` where `text` is the section body from `good.json`. Two drops of identical content produce identical hashes, so `collection.upsert(...)` overwrites rather than appending — this is the in-process embodiment of Hohpe and Woolf's idempotent-receiver pattern.

## Exercise 2 — Expected outputs

After `cp data/docs_inbox-templates/bad_invalid_json.json data/docs_inbox/bad_parse.json`:

```
ingestion.watcher: Quarantined bad_parse.json: invalid JSON: Expecting property name enclosed in double quotes at line N col M
```

`ls data/docs_inbox/failed/` then shows:

```
bad_parse.json
bad_parse.json.error.txt
```

`cat data/docs_inbox/failed/bad_parse.json.error.txt` reproduces the quarantine reason verbatim.

After `cp data/docs_inbox-templates/bad_missing_field.json data/docs_inbox/bad_schema.json`:

```
ingestion.watcher: Quarantined bad_schema.json: missing required fields: ['metadata']
```

The `failed/` directory now contains four artifacts (two `.json` + two `.error.txt`). The main inbox contains only `my_first_section.json` from Exercise 1. The two quarantine messages came from different layers of the watcher — the first from the `json.JSONDecodeError` exception handler at `src/ingestion/watcher.py:175-179`, the second from `validate_section` at `src/ingestion/watcher.py:78-103`. Two-stage validation surfaces the right diagnostic at the right layer.

### Exercise 2 stretch — failure summary one-liner

Expected paste-into-writeup shape (one row per failure, headers omitted for terseness):

```
bad_parse.json,invalid JSON: Expecting property name...,156
bad_schema.json,missing required fields: ['metadata'],89
```

A representative one-liner:

```python
from pathlib import Path
for p in sorted(Path('data/docs_inbox/failed').glob('*.json')):
    err = p.with_suffix(p.suffix + '.error.txt').read_text().strip()
    print(f"{p.name},{err},{p.stat().st_size}")
```

## Exercise 3 — Expected outputs

### Part A — Same-tag rebuild

`make migrate-blue-green` summary block (representative):

```
target_color=scikit_docs_blue
previous_color=scikit_docs  (legacy bootstrap)
recall@5=0.83
threshold=0.70
swapped=True
active_collection_path=data/ACTIVE_COLLECTION
```

`cat data/ACTIVE_COLLECTION` prints `scikit_docs_blue` (one line, no newline guarantees but the resolver tolerates both). The Python sanity check reports:

```
alias resolved to: scikit_docs_blue
chunk count: ~750
```

(Exact count depends on how many sections the current scikit-learn tag parses cleanly; expect 740-760.)

The `/query` call returns a grounded answer about the scikit-learn `Pipeline` API with sources from the blue collection — the gateway was not restarted, but the next `get_collection()` call resolved the alias to blue.

The `list_collections` walk shows at least:

```
scikit_docs → ~750 rows  (legacy, pre-migration)
scikit_docs_blue → ~750 rows  (new active color)
```

### Part B — Roll-forward to green

Second `make migrate-blue-green` summary:

```
target_color=scikit_docs_green
previous_color=scikit_docs_blue
recall@5=0.83  (same source tag, similar score)
threshold=0.70
swapped=True
```

`cat data/ACTIVE_COLLECTION` now prints `scikit_docs_green`. `list_collections` now shows three: `scikit_docs`, `scikit_docs_blue`, `scikit_docs_green`. The previous color (blue) is the natural rollback target.

### Part B stretch — failed gate

`uv run python scripts/migrate_blue_green.py --threshold 1.01 --keep-failed` summary:

```
target_color=scikit_docs_blue  (rebuilt; was the inactive color after Part B)
previous_color=scikit_docs_green
recall@5=1.000
threshold=1.01
swapped=False
note: --keep-failed retained scikit_docs_blue for forensics
```

The 12-row eval sample scores a perfect recall@5, so the gate can only be
forced to fail with a floor above 1.0 — `--threshold 1.01` does that. The
point is the mechanism (a failed gate leaves the alias untouched), not the
specific number.

`cat data/ACTIVE_COLLECTION` still prints `scikit_docs_green`. The freshly-rebuilt blue color exists in Chroma but the gateway is not querying it. The production property the alias mechanism exists to guarantee — *a failed gate cannot point the public alias at a bad collection* — held.

## Writeup answer — why in-place re-ingest fails

Expected 2-3 sentence answer:

> An in-place re-ingest first drops the live `scikit_docs` collection and then takes minutes to embed and upsert the rebuilt corpus. During that window the gateway queries a half-populated collection whose recall is unpredictable — every request lands on an index that is missing most of its rows. The blue/green pattern builds into a separate color that the gateway is not yet querying and only flips the alias after the new color has been measured against the same golden set; the `os.replace` swap is atomic, so the cutover takes microseconds instead of minutes, no gateway restart is required, and the previous color stays available as a rollback target.

## Verification — what to run as a grader

```bash
# After make setup + make load-data + make seed-difficulty:
make watch &           # Terminal 1 in practice; & for grading scripts only
WATCHER_PID=$!
sleep 3                # Let the watcher initialize

# Exercise 1
cp data/docs_inbox-templates/good.json data/docs_inbox/grade1.json
sleep 2
# Inspect watcher stdout for "Ingested grade1.json" line

# Exercise 2
cp data/docs_inbox-templates/bad_invalid_json.json data/docs_inbox/grade2a.json
cp data/docs_inbox-templates/bad_missing_field.json data/docs_inbox/grade2b.json
sleep 2
test -f data/docs_inbox/failed/grade2a.json.error.txt
test -f data/docs_inbox/failed/grade2b.json.error.txt

# Exercise 3 — alias swap
kill $WATCHER_PID
make migrate-blue-green
grep -q "scikit_docs_blue" data/ACTIVE_COLLECTION
make migrate-blue-green
grep -q "scikit_docs_green" data/ACTIVE_COLLECTION
```

## KNOWN-LIMITATIONs

- **No authored solution code.** These exercises are pure ops — no fenced code blocks in the instructions name a file to create or extend, so there is nothing to commit into `solution/src/` or `solution/scripts/` that does not already ship in the starter. Verification is empirical (run the make targets, inspect log lines, read the alias file) rather than `pytest`-driven for the exercise deliverables themselves; the starter's existing `tests/test_ingestion.py` covers the underlying watcher/alias/migrate units.
- **Exercise 3 requires real OpenAI embedding calls.** The blue/green migration re-embeds any cache-miss chunks, and on a cold-cache or new-tag run this costs ~$0.05-0.10 per migration. The expected-output recall numbers assume the cached `data/embedding_cache.jsonl` from a prior `make load-data` is present; on a clean clone, the first migration is slower and slightly more expensive.
- **scikit-learn checkout state.** `make migrate-blue-green` will `git clone` (or `git fetch`) the scikit-learn repo at `data/scikit-learn-cache/`. On the Workspace this is fast; on a clean grader machine without git installed, the migration will fail at the clone step rather than at the eval gate. The starter assumes git is available.
