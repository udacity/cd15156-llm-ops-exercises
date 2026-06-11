# Automate the ScikitDocs Pipeline (watchdog ingest + blue/green index swap)

## Setup

This starter is the ScikitDocs RAG app — a Q&A assistant for the scikit-learn library — with the full instrumentation stack already wired: prompt loader and Jinja templates, Chroma vector store, RAG pipeline (`run_pipeline`), Phoenix tracing, RAGAS evaluation harness, cost monitoring, semantic answer cache, FastAPI gateway, guardrails, A/B testing, RAGOps watcher (`src/ingestion/` + `scripts/start_watcher.py`), blue/green migration (`src/ingestion/migrate.py` + `scripts/migrate_blue_green.py`), and a streaming endpoint. In this module you will exercise the RAGOps surface end to end without modifying any source: drive the watcher through happy-path and quarantine flows, then run a blue/green re-ingest and verify the alias swaps atomically. There is no new code to author — the work is operational reasoning plus shell + curl drills against existing seams.

Bring up the corpus before you start:

```bash
make setup                    # uv sync
cp .env.example .env          # add your OPENAI_API_KEY (or Vocareum voc- key)
make load-data                # ~45–60s cold, ~5s warm; ~$0.10 in embeddings
make seed-difficulty          # tags golden_set.csv rows simple|complex
```

Smoke-check the pipeline before any RAGOps work:

```bash
uv run python -c "from src.pipeline import run_pipeline; print(run_pipeline('What kernel does SVC use by default?').answer[:80])"
```

If that returns a grounded answer about `rbf`, you are ready. Watch the recorded demo first, then work through the three exercises in order.

---

# Exercises — Drive the Watcher End to End, Then Atomically Swap an Index

Three exercises. The first wires the watcher into a live round trip — start `make watch`, drop a good section JSON, see it ingest, query it back through the FastAPI gateway. The second exercises the quarantine path with deliberately malformed files and walks the failure forensics. The third runs a blue/green re-ingest against the inactive color and verifies the alias swaps atomically — this is the migration analog you reason about, then run, then verify. Each exercise has a `Success Criteria` block that names what "done" looks like. Common Pitfalls are at the end and worth a skim before you start.

You will not modify the ScikitDocs starter source tree. Everything in this module operates on existing seams — the inbox directory at `data/docs_inbox/`, the file-watcher CLI at `scripts/start_watcher.py`, the migration script at `scripts/migrate_blue_green.py`, and the section JSON templates at `data/docs_inbox-templates/`. Plan for roughly twenty minutes total, weighted toward Exercise 3 where the alias-swap reasoning happens.

## Pre-flight check

You should have `make verify` passing and `make load-data` reporting the corpus is loaded. The `.env` file carries `OPENAI_API_KEY` and (on Vocareum) `OPENAI_BASE_URL` set to `https://openai.vocareum.com/v1`. If you are unsure which environment you are on:

```
uv run python -c "from src.config import settings; print(repr(settings.openai_base_url))"
```

You want `''` on direct OpenAI or `'https://openai.vocareum.com/v1'` on Vocareum. Any other value will fail with a confusing 401 the first time the watcher tries to embed an incoming section — fix the `.env` before continuing.

Confirm the corpus is loaded and the alias is in its default pre-migration state:

```
uv run python -c "
from src import store
from src.ingestion import read_active_collection, ACTIVE_COLLECTION_PATH
print('chunk count:', store.get_collection().count())
print('active collection (from alias file):', read_active_collection())
print('alias file exists:', ACTIVE_COLLECTION_PATH.exists())
"
```

You want a chunk count of about 755 (the value matches whatever `make load-data` and `make seed-difficulty` together produced), an active collection of `scikit_docs` (the literal alias name, which the resolver returns when the file is missing), and `alias file exists: False`. That last line is the bootstrap state — Exercise 3 is what creates the file for the first time.

You will work in two terminals throughout the exercise. The first runs the watcher; the second is where you drop files and run curl. Open both before you start.

## Exercise 1 — Drop a good section, watch it ingest

The pattern is: watcher in one terminal, file-drop in the other, then a query through the live gateway to confirm the index reflects the new section. The point of Exercise 1 is to see the round trip work end to end before the next two exercises introduce failure modes.

### What to do

1. In terminal one, start the watcher with verbose logging:

   ```
   uv run python scripts/start_watcher.py --log-level DEBUG
   ```

   You should see the line `[watcher] watching data/docs_inbox for new section JSONs` and then nothing. The watcher is idle until something lands in the inbox.

2. In a separate shell, start the gateway so the query path is live:

   ```
   make serve
   ```

   Give it five to ten seconds to load the embedder and open the Chroma collection. You will know it is ready when the log shows `Uvicorn running on http://0.0.0.0:8080`.

3. In terminal two, copy the good template into the inbox under a fresh filename:

   ```
   cp data/docs_inbox-templates/good.json \
      data/docs_inbox/my_first_section.json
   ```

   Within roughly one second, the watcher log in terminal one should print:

   ```
   ingestion.watcher: Ingested my_first_section.json → modules.inbox_demo.intro#... (active=scikit_docs)
   ```

   The `active=scikit_docs` portion is the alias resolver reporting itself — since the alias file does not exist yet, writes land in the original `scikit_docs` collection from before the alias mechanism existed. Exercise 3 is where this changes to a color name.

4. Confirm the new section is queryable through the gateway. From terminal two:

   ```
   curl -X POST http://localhost:8080/query \
     -H "Content-Type: application/json" \
     -d '{"question": "How does the scikit-learn docs inbox watcher work?"}'
   ```

   The response body should mention the inbox watcher by name, cite the content-hashed chunk id detail from the template, and include `modules.inbox_demo.intro` somewhere in the `sources` array. The exact ranking depends on the rest of the corpus, but the new section should land in the top five hits.

5. Look at the inbox in terminal two. The file you dropped is still there — the watcher does not delete successful ingestions. That is by design: the inbox is the audit trail, and a separate retention policy (a cron job, a manual cleanup) decides when old files go away. In the AWS production form from the recorded demo's sidebar, the S3 bucket plays this role and an S3 Lifecycle policy is what eventually moves old objects to Glacier.

### Success Criteria

- The watcher log shows the `Ingested my_first_section.json` line within roughly two seconds of the `cp` command.
- The query against `/query` returns an answer that names the inbox watcher and lists `modules.inbox_demo.intro` (or its content-hashed chunk id) in the cited sources.
- The dropped file remains in `data/docs_inbox/` (not moved to `failed/`).
- The watcher process is still running and ready for the next exercise.

### Stretch

Drop the same `good.json` twice in quick succession under two different filenames. Confirm the watcher logs `Ingested ...` for both, then verify only one row was added to the collection:

```
uv run python -c "
from src import store
c = store.get_collection()
matches = c.get(where={'section_title': 'Inbox Demo Intro'})
print('row count:', len(matches['ids']))
print('chunk id:', matches['ids'][0])
"
```

You should see `row count: 1`. The chunk-id content hash made the two drops resolve to the same Chroma row; the second upsert overwrote the first. This is Hohpe and Woolf's idempotent-receiver property at the code seam.

## Exercise 2 — Trigger quarantine on bad input

The `bad_invalid_json.json` template has a trailing comma that makes it syntactically invalid JSON; `bad_missing_field.json` is well-formed JSON missing the `metadata` block the validator requires. Both should land in `data/docs_inbox/failed/` with a sibling `.error.txt`, each at a different layer of the watcher's two-stage check.

### What to do

1. Keep the watcher running from Exercise 1. In terminal two:

   ```
   cp data/docs_inbox-templates/bad_invalid_json.json \
      data/docs_inbox/bad_parse.json
   ```

2. Within roughly one second, the watcher log should print a warning along these lines:

   ```
   ingestion.watcher: Quarantined bad_parse.json: invalid JSON: Expecting property name enclosed in double quotes at line N col M
   ```

   The exact line and column depend on the editor's line-ending convention, but the failure reason is the trailing comma. The watcher caught the parse failure before it ever got to schema validation — `json.loads` raised, the exception handler at `src/ingestion/watcher.py:175-179` formatted the reason, and `_quarantine` moved the file.

3. Confirm the quarantine moved the file and recorded the reason. In terminal two:

   ```
   ls data/docs_inbox/failed/
   cat data/docs_inbox/failed/bad_parse.json.error.txt
   ```

   You should see two files (`bad_parse.json` and `bad_parse.json.error.txt`), and the `.error.txt` reason should match the quarantine log line.

4. Confirm the inbox no longer has the bad file:

   ```
   ls data/docs_inbox/*.json
   ```

   `bad_parse.json` should not be in the list — it was moved (not copied) into `failed/`, so the inbox is clean and the watcher does not retry the same failure on every restart. `recursive=False` on the observer (see `start_observer` at `src/ingestion/watcher.py:235-244`) means subdirectories of the inbox are not watched, so files in `failed/` are not re-ingested.

5. Walk the second failure mode. Drop the missing-field template:

   ```
   cp data/docs_inbox-templates/bad_missing_field.json \
      data/docs_inbox/bad_schema.json
   ```

   Within roughly one second, the watcher should log a new quarantine — this time the failure reason is the schema validator, not the JSON parser:

   ```
   ingestion.watcher: Quarantined bad_schema.json: missing required fields: ['metadata']
   ```

   `validate_section` at `src/ingestion/watcher.py:78-103` produced the message, and the failure mode is now visible at a different layer than the JSON parse failure. Two-stage validation matters because the right fix differs — a JSON parse error is a producer-side authoring bug, a missing-field error is a schema-contract mismatch that probably means the upstream pipeline was rebuilt without honoring the watcher's `REQUIRED_METADATA_FIELDS` contract.

### Success Criteria

- The first `cp` lands the bad file in `data/docs_inbox/failed/` with a sibling `.error.txt` whose content names a JSON parse error.
- The second `cp` lands the schema-violating file in `failed/` with a sibling `.error.txt` whose content names the missing `metadata` field.
- The `failed/` directory retains the prior failure artifacts as the audit trail.
- The watcher is still running and the main inbox is empty of any `.json` files except the one from Exercise 1.

### Stretch

Write a Python one-liner that walks `data/docs_inbox/failed/` and prints a CSV-shaped summary of the failures — filename, error reason, file size. This is the development-time analog of the failure-monitoring dashboard you would build in production against SQS DLQ message attributes; doing it once locally gives you the data-shape intuition for the real thing.

## Exercise 3 — Run a blue/green migration and verify the alias swap

Two parts. Part A runs the migration end-to-end against the same corpus tag (the rebuild form of blue/green — same source, new collection, atomic swap). Part B reasons about what changes when the source tag is actually different (the version-upgrade form), and verifies the alias mechanism itself with a manual swap-back. This is the migration mechanism you reason about *and* run; the alias swap is the load-bearing operational property at the heart of the pattern.

Two ideas worth holding in mind before you run it. First, the alias swap is not the same operation as "delete and rebuild." A delete-and-rebuild against the live `scikit_docs` collection serves partial results for the minutes the rebuild takes — every query during that window hits a half-populated index whose recall is unpredictable, and there is no clean way to roll back if the rebuild produces a worse collection than what you had. The blue/green pattern fixes both problems by building into a *separate* collection that the gateway is not querying, then flipping the alias only after the new collection has been measured against the same golden set the old one was measured against. Second, the eval gate is the load-bearing detail. Without it, the swap is just a fast way to ship a regression. With it, the swap is a release procedure — measurement first, swap second, and a knob (the recall floor) that decides what "good enough" means.

### Part A — Same-tag rebuild and atomic swap

The point is to see the alias file appear, the cutover happen atomically, and the gateway start returning hits from the new color without any restart.

1. Stop the watcher from Exercises 1 and 2 (Ctrl+C in terminal one) so the migration has the embedding cache to itself. Leave the gateway running. Then run:

   ```
   make migrate-blue-green
   ```

   You will see the script clone scikit-learn (or reuse the cached clone), parse the RST tree, embed (most chunks cache-hit), and upsert into `scikit_docs_blue` — the bootstrap target on a starter where the alias file does not yet exist. The next phase runs `recall_at_k` against the first twelve rows of `data/golden_set.csv`, and on a pass calls `swap_alias`. The summary block at the end prints `swapped=True`, the recall number, and the path to the new `data/ACTIVE_COLLECTION` file.

2. Verify the alias file was created and points at blue:

   ```
   cat data/ACTIVE_COLLECTION
   uv run python -c "
   from src import store
   c = store.get_collection()
   print('alias resolved to:', c.name)
   print('chunk count:', c.count())
   "
   ```

   You should see `scikit_docs_blue` in the file, the same color reported as the collection name, and a chunk count close to 750 (the seed-difficulty rows are not part of the migration — the blue color is a clean rebuild of the corpus alone).

3. Run a query against the gateway from terminal two without restarting it:

   ```
   curl -X POST http://localhost:8080/query \
     -H "Content-Type: application/json" \
     -d '{"question": "How do I use scikit-learn pipelines?"}'
   ```

   The answer should reference the scikit-learn `Pipeline` API. The gateway picked up the new color without restart because `get_collection` resolves the alias on every call — there is no in-process cache of the collection handle past the Chroma client itself. That property is what makes the cutover zero-downtime: no client reconnects, no warmup, no cache invalidation.

4. Confirm the previous color still exists (or, in this first run, the legacy `scikit_docs` collection from before the migration). From terminal two:

   ```
   uv run python -c "
   from src import store
   client = store._client()
   for c in client.list_collections():
       print(c.name, '→', c.count(), 'rows')
   "
   ```

   You should see at least two collections — `scikit_docs_blue` (the new active color, populated by the migration) and `scikit_docs` (the original collection from before the alias mechanism existed). The old collection sits there as a fallback you could roll back to with a manual `swap_alias` — but the cleanest rollback in the blue/green model is a *second* migration that builds green and swaps to it, not a flip back to a stale collection. Part B walks this.

### Part B — Inspect the cutover, then a manual roll-forward to green

1. Read `data/CORPUS_VERSION`:

   ```
   cat data/CORPUS_VERSION
   ```

   The `scikit_learn_tag` line is what `make migrate-blue-green` would re-ingest by default. In production, the version-upgrade scenario is: a new scikit-learn release ships, you bump `corpus.SCIKIT_LEARN_TAG` in `src/corpus.py` (or pass `--tag 1.6.0` to the migration script), and the next `make migrate-blue-green` ingests the new version into the green color, evaluates it against the same golden set, and swaps if the recall floor holds. The same script, the same alias, the same eval gate — only the source tag changes.

2. Force a roll-forward by running the migration a second time. Because the alias now names `scikit_docs_blue`, the next migration targets `scikit_docs_green`:

   ```
   make migrate-blue-green
   ```

   The summary should report `target_color=scikit_docs_green`, `previous_color=scikit_docs_blue`, and `swapped=True`. Re-read the alias file:

   ```
   cat data/ACTIVE_COLLECTION
   ```

   It now says `scikit_docs_green`. The gateway started serving hits from green the moment `os.replace` returned — no warmup, no restart, no in-process cache to bust. That is the operational property the whole pattern exists for.

3. Stretch — simulate a regression. Edit `scripts/migrate_blue_green.py`'s default threshold by passing `--threshold 0.99` on the command line: a value the corpus genuinely cannot hit on twelve rows. Run:

   ```
   uv run python scripts/migrate_blue_green.py --threshold 0.99 --keep-failed
   ```

   The build runs, the gate fails, and the summary reports `swapped=False` with the recall number printed alongside the threshold. The alias stays where it was (still pointing at green from Step 2). The `--keep-failed` flag leaves the freshly-built color in Chroma for forensics; without it, the script drops the failed color so the next attempt starts clean. Either way, the production property held: a failed gate cannot point the public alias at a bad collection.

### Success Criteria

- Part A: `make migrate-blue-green` completes with `swapped=True`, `data/ACTIVE_COLLECTION` exists and names `scikit_docs_blue`, and a `/query` call without a gateway restart returns hits.
- Part B: a second `make migrate-blue-green` swaps to `scikit_docs_green` and the file is updated. A migration with `--threshold 0.99` reports `swapped=False` and leaves the alias on the previously-swapped color.
- You can articulate, in two or three sentences, why an in-place re-ingest of the live `scikit_docs` collection would have served partial results during the rebuild window and why the alias swap fixes that without needing a load-balancer or gateway restart.

## Common Pitfalls

- **Forgetting to start the watcher.** Drops into the inbox sit there silently and nothing ingests them. The symptom is "I dropped the file, but the query does not find the section." Confirm `make watch` is running in a terminal you can see before you drop files.
- **Vocareum base-URL not set.** The watcher calls the embedding API the first time a section lands. If `.env` has `OPENAI_API_KEY` but no `OPENAI_BASE_URL` on a Vocareum-issued key, the call fails with a 401 and the file ends up in `failed/` with a confusing error. The setup check at the top of this module catches it before you start.
- **Running the watcher and the migration concurrently.** The migration drops and rebuilds the inactive color; while it runs, the watcher's content-hashed upserts into the *active* color are fine — but the embedding cache file is appended-to by both, and a torn write would corrupt one entry. The watcher's `_embed_one` uses the same cache helpers and serializes per-call, but the safer pattern is to stop the watcher for the migration window. Exercise 3 step 1 names this.
- **Editing files in-place inside the inbox.** Watchdog fires `modified` events on in-place edits (the handler only listens for `created` and `moved`, so most edits are silently ignored on Linux — but on macOS/FSEvents you may see surprises). The atomic pattern is the same as for `cp`: write the file under a different name, then `mv` it into the inbox.
- **Network filesystems silencing events.** NFS and SMB do not emit native filesystem events to the kernel layer watchdog uses, so the default `Observer` will sit silent. The mitigation is `PollingObserver` from `watchdog.observers.polling`, which stat-loops the directory. The Workspace runs on local disk, so the default is the right choice here.
- **Forgetting to invalidate the semantic cache after a migration.** The semantic answer cache may hold paraphrased answers that referenced the prior color. A blunt eviction with `clear()` is the simplest fix; targeted invalidation by source citation is the rigorous one. Either way, the cache and the alias are coupled — a migration that swaps the alias should also bust the cache, or the first cohort of queries serves answers built against the previous color.
- **Treating `data/ACTIVE_COLLECTION` as scratch.** Deleting the file mid-experiment reverts the alias to legacy mode (the resolver falls back to the literal `scikit_docs`), which silently re-routes the gateway to the original collection from before the migration. If you want to roll back, run another migration; do not edit the file by hand.
- **Setting `--threshold` too high.** A reasonable production starting point is the *previous* migration's recall@5, minus a small tolerance — not a hard 0.85 floor. The default `0.70` in `migrate_blue_green` matches the smoke gate; tightening it should be a deliberate calibration decision, not a typo.

What you have at the end. A watcher that ingests new doc sections in roughly a second, a quarantine path that preserves failures for debugging, a blue/green migration that atomically cuts the gateway over to a freshly-built and gate-passed collection, and the operational reasoning to handle the three re-ingest events that drive a re-index — drift, embedding-model migration, and source-version upgrade. The watchdog form runs locally; the AWS form from the recorded demo's sidebar is the same pattern at production scale; the alias swap is the load-bearing property that lets either form carry the index between versions without a serving window.
