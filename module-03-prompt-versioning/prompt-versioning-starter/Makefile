.PHONY: help setup clean serve serve-proxy load-data seed-difficulty smoke-gate show-traces seed-traces seed-cost-log cost-dashboard eval eval-topk-sweep watch migrate-blue-green ab-analyze test verify consistency-check

# Make `src` importable for scripts run via `uv run python scripts/...`.
# (Pytest gets this from `pythonpath = ["."]` in pyproject.toml; scripts don't.)
export PYTHONPATH := .

# Force uv to copy packages from its cache into the venv instead of hardlinking.
# uv's default is hardlink, which is faster on a single filesystem but emits a
# "warning: failed to hardlink file from ..." line and falls back to copy when
# the cache and venv live on different mounts (common on the Vocareum
# Workspace). Mirrors capstone Makefile.
export UV_LINK_MODE := copy

# Cap RAGAS executor concurrency (filled in by Module 11 — placeholder here
# so the variable exists at scaffold time and consistency-check passes).
EVAL_MAX_WORKERS ?= 1

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies with uv
	uv sync

clean: ## Remove generated artifacts (venv + corpus caches) for a from-scratch setup/load-data
	rm -rf .venv data/chroma data/phoenix data/scikit-learn-cache data/embedding_cache.jsonl data/CORPUS_VERSION

serve: ## Run FastAPI development server (Module 18 fills src/gateway/app.py)
	# --reload-dir scopes the file-watcher to src/ only.
	uv run uvicorn src.gateway.app:app --reload --reload-dir src --port 8080

serve-proxy: ## Run FastAPI dev server for Udacity Workspace (/proxy/{port} prefix)
	PHOENIX_HOST_ROOT_PATH=/proxy/6006 \
	uv run uvicorn src.gateway.app:app --reload --reload-dir src --port 8080 --root-path /proxy/8080

load-data: ## Ingest scikit-learn docs into Chroma (fills scripts/load_data.py)
	uv run python scripts/load_data.py

seed-difficulty: ## Upsert 8 deliberately-confusing chunks — run after load-data
	uv run python scripts/seed_difficulty.py

smoke-gate: ## Wave-4 smoke RAG gate — recall@5 ≥ 0.7 on 3 representative queries
	uv run python scripts/smoke_gate.py

show-traces: ## Export recent Phoenix traces as markdown (Module 09 — rubric §7 backstop)
	uv run python scripts/show_traces.py

seed-traces: ## Fire a 5-query pack and write the rubric §7 evidence file at data/trace_evidence.md
	uv run python scripts/seed_traces.py

seed-cost-log: ## Append 50 synthetic entries to data/cost_log.jsonl (Module 13) — idempotent, $0 cost
	uv run python scripts/seed_cost_log.py

cost-dashboard: ## Standalone /cost-dashboard on localhost:8080 (Module 13) — Module 18 mounts it on the gateway
	uv run python -m src.cost.dashboard

eval: ## Run RAGAS evaluation against the golden set (Module 11)
	uv run python scripts/run_eval.py --max-workers=$(EVAL_MAX_WORKERS)

eval-topk-sweep: ## Sweep top_k ∈ {3,5,10} and emit a markdown table (Module 11)
	uv run python scripts/eval_topk_sweep.py --max-workers=$(EVAL_MAX_WORKERS)

watch: ## Start the docs-inbox watcher (Module 24) — ingests JSONs dropped into data/docs_inbox/
	uv run python scripts/start_watcher.py

migrate-blue-green: ## Re-ingest the corpus into the inactive color and swap the alias on pass (Module 24)
	uv run python scripts/migrate_blue_green.py

ab-analyze: ## Read data/ab_log.jsonl and print chi-squared + per-variant cost/latency (Module 22)
	uv run python scripts/ab_analyze.py

test: ## Run the unit + smoke test suite
	uv run pytest tests/ -q

verify: ## Run lint + tests (placeholder; per-module checks added by downstream modules)
	uv run pytest tests/ -q
