# Shared environment for all Course 2 exercise modules.
#
# All 12 modules (24 starter/solution roots) share ONE dependency closure,
# pinned in requirements.txt — the same file the Udacity Workspace image is
# built from.
#
#   - Udacity Workspace (Vocareum): dependencies are preinstalled in the
#     image. Nothing to run here; each module's `make setup` only verifies
#     the environment.
#   - Local machine: run `make setup` here ONCE to build the single shared
#     venv at exercises/.venv. Module Makefiles then use it via
#     `uv run --no-project` — no per-module .venv is ever created.

.PHONY: help setup clean

# Udacity Workspace (Vocareum) marker: sessions mount the course repo at
# /workspace. Overridable for testing (make setup WORKSPACE=/some/dir).
WORKSPACE ?= /workspace

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Provision the shared env — verify first; install only on a local machine
	@uv run --no-project python -c "import fastapi, uvicorn, pytest, chromadb, openai, ragas, phoenix, llm_guard, watchdog" 2>/dev/null \
		&& echo "Environment already provides all dependencies — nothing to install." \
		|| { if [ -d "$(WORKSPACE)" ]; then \
		       echo "This Udacity Workspace's image should provide all dependencies, but some are missing."; \
		       echo "NOT installing here: a venv would exceed the Workspace storage allowance."; \
		       echo "Please report the missing packages to Udacity support."; \
		       exit 1; \
		     else \
		       uv venv .venv --python 3.12 && uv pip install -r requirements.txt; \
		     fi; }

clean: ## Remove the shared venv
	rm -rf .venv
