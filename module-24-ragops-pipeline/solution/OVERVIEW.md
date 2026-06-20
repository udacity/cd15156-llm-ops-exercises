---
module_number: 24
module_title: "Automate the ScikitDocs Pipeline (watchdog ingest + blue/green index swap)"
slug: ragops-pipeline
unit_type: overview-video-script
target_minutes: "5-7"
word_count: 720
---

# Module 24 Overview Video: RAGOps Pipeline (Watcher + Blue/Green Swap)

> Recording aid for a 5-7 min overview (read pace ~100 wpm). Read all prose aloud, top to bottom. `*[Stage: ...]*` lines are screen directions, not spoken. Not learner-facing.

## 0. VS Code setup (before you hit record)

Open the files you will show on camera:

- `../ragops-pipeline-starter/DEMO.md`
- `../ragops-pipeline-starter/INSTRUCTIONS.md`
- `../ragops-pipeline-starter/INTERFACES.md`
- `../ragops-pipeline-starter/src/ingestion/watcher.py`
- `../ragops-pipeline-starter/src/ingestion/alias.py`
- `../ragops-pipeline-starter/src/ingestion/migrate.py`
- `../ragops-pipeline-starter/src/store.py`
- `../ragops-pipeline-starter/scripts/start_watcher.py`
- `../ragops-pipeline-starter/scripts/migrate_blue_green.py`
- `../ragops-pipeline-starter/data/docs_inbox-templates/good.json`
- `../ragops-pipeline-starter/data/golden_set.csv`

Files and why each is on screen:
- `DEMO.md`: the codebase walkthrough; covers the watcher, the blue/green swap, and the AWS sidebar.
- `INSTRUCTIONS.md`: the three operational exercises; keep it open to point at each Success Criteria block.
- `INTERFACES.md`: the frozen signatures, including how `get_collection("scikit_docs")` resolves the alias.
- `src/ingestion/watcher.py`: the file-watcher; `validate_section`, `chunk_id_for`, and `_quarantine` all live here.
- `src/ingestion/alias.py`: the two color names and `swap_alias`, the atomic rename at the heart of the cutover.
- `src/ingestion/migrate.py`: `migrate_blue_green`, the four-step rebuild, eval-gate, and swap orchestrator.
- `src/store.py`: where `get_collection` resolves the public alias to the active color on every call.
- `scripts/start_watcher.py`: the watcher lifecycle the `make watch` target wraps.
- `scripts/migrate_blue_green.py`: the migration command line, including the `--threshold` knob.
- `data/docs_inbox-templates/good.json`: the section JSON you drop in Exercise 1; the schema contract made visible.
- `data/golden_set.csv`: the questions the eval gate scores recall against before it allows a swap.

## 1. Intro  (~45-60s)

*[Stage: DEMO.md on screen.]*

In this module you run a retrieval-augmented generation pipeline, or RAG, like an operator, not a coder. There's no new code to write. You drive the live system through three things: a file-watcher that ingests new documents, a quarantine path for bad input, and a blue/green index swap that updates the corpus with no downtime.

This is the operations layer that keeps a RAG app fresh. Documents change, and the index has to follow them safely. The skill is reasoning about the pipeline, then running and verifying the drills.

## 2. Topic overview  (~60-90s)

*[Stage: bring src/ingestion/watcher.py and src/ingestion/alias.py forward.]*

The pipeline has three roles: a producer, a queue, and a consumer. A document section lands in an inbox folder. A watcher notices it and ingests it into the vector store. The load-bearing property is on the consumer side, and it's called idempotency.

Here's what that means. The chunk id is built from a hash of the text, so the same payload always produces the same id. Re-dropping a file overwrites the existing row instead of adding a duplicate. That's an upsert, not an append.

*[Stage: point to validate_section in src/ingestion/watcher.py.]*

The watcher also validates every file. Bad input doesn't get silently dropped. It moves to a separate quarantine folder with a note explaining why, so the failure is preserved for someone to read later.

The common misconception is that updating a live index means editing it in place. It doesn't. You build a fresh copy alongside the live one, then flip a pointer to it. That's blue/green.

## 3. Exercise call-outs

### Exercise 1: Drop a good section, watch it ingest

*[Stage: switch to INSTRUCTIONS.md, Exercise 1; keep good.json visible.]*

The warm-up here is the round trip. You start the watcher in one terminal, the gateway in another, then drop a good section file into the inbox. Within about a second, the watcher ingests it and the gateway can answer about it.

Here's what to watch out for. The dropped file stays in the inbox after a successful ingest. That's on purpose: the inbox is your audit trail, not a delete-after-processing queue. In the stretch task, drop the same file twice under different names. You'll see two ingest lines but only one row, because the content hash made both drops resolve to the same id. That's idempotency at the code seam.

To complete this exercise, confirm the gateway answers the query by citing the inbox watcher — and check that the dropped file is still sitting in the inbox, not moved to the failed folder.

### Exercise 2: Trigger quarantine on bad input

*[Stage: point to _quarantine in src/ingestion/watcher.py.]*

This exercise is about failure handling. You drop two broken files. One is invalid JSON. The other is valid JSON missing a required field. The validator catches them at two stages.

Here's the watch-out, and it's the whole point. Bad input is never silently discarded. Each broken file moves to a quarantine folder with a sibling error file naming the reason. The two stages matter because the fix differs. A parse error is an authoring bug upstream. A missing field means the producer changed shape and broke the contract.

To complete this exercise, get both broken files into the failed folder — each with its own error note — while the inbox itself stays untouched.

### Exercise 3: Run a blue/green migration and verify the alias swap

*[Stage: open src/ingestion/alias.py, point to swap_alias; have migrate.py ready.]*

This is the heart of the module. You rebuild the corpus into an inactive color, score it against the golden set, and swap the alias only if it passes. The gateway picks up the new color with no restart, because the store resolves the alias on every query.

One watch-out on the eval gate. It's about the mechanism, not the number. Recall on this small golden set is essentially perfect, so the stretch sets the threshold above what's achievable. That forces a no-swap decision, so you watch the gate refuse a release.

To complete this exercise, verify the alias file exists and names a color after the first migration, watch the second migration roll forward to the other color, and confirm that a gate failure leaves the alias exactly where it was.

## 4. Key insights  (~30-45s)

*[Stage: point to get_collection in src/store.py.]*

Three takeaways. First, idempotency is what makes re-ingest safe: same payload, same id, an overwrite instead of a duplicate. Second, failures go to quarantine with a reason, never into a silent void, so the next file isn't blocked. Third, the blue/green swap is a release procedure: measure first, swap second, and the alias flip is atomic, so there's no window where the index is half-built.

## 5. Outro  (~20-30s)

*[Stage: return to DEMO.md, scroll to the AWS sidebar.]*

That's RAGOps: a watcher that ingests safely, a quarantine path that preserves failures, and a blue/green swap gated on quality. The same pattern scales to the cloud form in the demo's sidebar: a bucket, a function, a managed index. Same shape, bigger blast radius. See you in the next one.
