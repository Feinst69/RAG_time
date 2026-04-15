# ─────────────────────────────────────────────────────────────────────────────
# Qdrant RAG – Makefile
#
# Prerequisites: uv, docker
# All python commands go through `uv run` to stay inside the managed venv.
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help install \
        qdrant-up qdrant-down qdrant-logs \
        create-collection recreate-collection \
        index index-sample \
        search test-search \
        demo demo-full \
        info \
        lint fmt

# ── configurable defaults ────────────────────────────────────────────────────
QDRANT_URL      ?= http://localhost:6333
CSV             ?= data/customer_support_tickets_200k.csv
BATCH_SIZE      ?= 64
CANDIDATES      ?= 100
TOP_K           ?= 10
QUERY           ?= payment issue not resolved
QDRANT_STORAGE  ?= $(PWD)/qdrant_storage

# ─────────────────────────────────────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────────────────────────────────────
help:          ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*##"}{ printf "  \033[36m%-26s\033[0m %s\n", $$1, $$2 }'

# ─────────────────────────────────────────────────────────────────────────────
# Dependencies
# ─────────────────────────────────────────────────────────────────────────────
install:       ## Install all dependencies with uv
	uv sync

# ─────────────────────────────────────────────────────────────────────────────
# Qdrant server (Docker)
# ─────────────────────────────────────────────────────────────────────────────
qdrant-up:     ## Start the Qdrant container (restarts it if already exists)
	@mkdir -p $(QDRANT_STORAGE)
	@if docker ps -a --format '{{.Names}}' | grep -q '^qdrant$$'; then \
	  echo "Container 'qdrant' already exists – restarting …"; \
	  docker restart qdrant; \
	else \
	  docker run -d \
	    --name qdrant \
	    -p 6333:6333 \
	    -p 6334:6334 \
	    -v $(QDRANT_STORAGE):/qdrant/storage:z \
	    qdrant/qdrant; \
	fi
	@echo "Waiting for Qdrant to be ready …"
	@until curl -sf $(QDRANT_URL)/healthz > /dev/null 2>&1; do \
	  printf '.'; sleep 1; \
	done
	@echo " ready."
	@echo "Qdrant running at $(QDRANT_URL)  (dashboard: http://localhost:6333/dashboard)"

qdrant-down:   ## Stop and remove the Qdrant container
	docker stop qdrant && docker rm qdrant

qdrant-logs:   ## Tail Qdrant container logs
	docker logs -f qdrant

# ─────────────────────────────────────────────────────────────────────────────
# Collection management
# ─────────────────────────────────────────────────────────────────────────────
create-collection:    ## Create collection + payload indexes (no-op if it exists)
	uv run python -m qdrant_rag.rag.cli create-collection --url $(QDRANT_URL)

recreate-collection:  ## Drop + recreate the collection (destroys all data!)
	uv run python -m qdrant_rag.rag.cli create-collection --url $(QDRANT_URL) --recreate

# ─────────────────────────────────────────────────────────────────────────────
# Indexing
# ─────────────────────────────────────────────────────────────────────────────
index:         ## Index the full CSV (CSV=path, BATCH_SIZE=N)
	uv run python -m qdrant_rag.rag.cli index \
	  --csv $(CSV) \
	  --url $(QDRANT_URL) \
	  --batch-size $(BATCH_SIZE)

index-sample:  ## Index first 500 rows (quick smoke-test)
	uv run python -m qdrant_rag.rag.cli index \
	  --csv $(CSV) \
	  --url $(QDRANT_URL) \
	  --batch-size $(BATCH_SIZE) \
	  --limit 500

# ─────────────────────────────────────────────────────────────────────────────
# Search  (override QUERY, TOP_K, CANDIDATES, or any filter flag)
#
# Examples:
#   make search QUERY="subscription cancelled" PRIORITY=Urgent
#   make search QUERY="crash on upload" PRODUCT="Web Portal" LANGUAGE=French
#   make search QUERY="payment failed" SLA_BREACHED=true SEGMENT="Corporate"
# ─────────────────────────────────────────────────────────────────────────────

# Optional filter flags (empty by default → no filter applied)
PRIORITY        ?=
PRODUCT         ?=
CATEGORY        ?=
CHANNEL         ?=
REGION          ?=
LANGUAGE        ?=
SEGMENT         ?=
OS              ?=
BROWSER         ?=
PAYMENT_METHOD  ?=
ESCALATED       ?=
SLA_BREACHED    ?=
AGE_MIN         ?=
AGE_MAX         ?=
CREATED_FROM    ?=
CREATED_TO      ?=

# Build optional filter flags conditionally
_FILTER_FLAGS :=
ifneq ($(PRIORITY),)
  _FILTER_FLAGS += --priority "$(PRIORITY)"
endif
ifneq ($(PRODUCT),)
  _FILTER_FLAGS += --product "$(PRODUCT)"
endif
ifneq ($(CATEGORY),)
  _FILTER_FLAGS += --category "$(CATEGORY)"
endif
ifneq ($(CHANNEL),)
  _FILTER_FLAGS += --channel "$(CHANNEL)"
endif
ifneq ($(REGION),)
  _FILTER_FLAGS += --region "$(REGION)"
endif
ifneq ($(LANGUAGE),)
  _FILTER_FLAGS += --language "$(LANGUAGE)"
endif
ifneq ($(SEGMENT),)
  _FILTER_FLAGS += --segment "$(SEGMENT)"
endif
ifneq ($(OS),)
  _FILTER_FLAGS += --os "$(OS)"
endif
ifneq ($(BROWSER),)
  _FILTER_FLAGS += --browser "$(BROWSER)"
endif
ifneq ($(PAYMENT_METHOD),)
  _FILTER_FLAGS += --payment-method "$(PAYMENT_METHOD)"
endif
ifneq ($(ESCALATED),)
  _FILTER_FLAGS += --escalated $(ESCALATED)
endif
ifneq ($(SLA_BREACHED),)
  _FILTER_FLAGS += --sla-breached $(SLA_BREACHED)
endif
ifneq ($(AGE_MIN),)
  _FILTER_FLAGS += --age-min $(AGE_MIN)
endif
ifneq ($(AGE_MAX),)
  _FILTER_FLAGS += --age-max $(AGE_MAX)
endif
ifneq ($(CREATED_FROM),)
  _FILTER_FLAGS += --created-from $(CREATED_FROM)
endif
ifneq ($(CREATED_TO),)
  _FILTER_FLAGS += --created-to $(CREATED_TO)
endif

search:        ## Hybrid search with optional metadata filters (see header comments)
	uv run python -m qdrant_rag.rag.cli search \
	  "$(QUERY)" \
	  --url $(QDRANT_URL) \
	  --top-k $(TOP_K) \
	  --candidates $(CANDIDATES) \
	  $(_FILTER_FLAGS)

# ─────────────────────────────────────────────────────────────────────────────
# Test searches  (fixed queries that exercise filters)
# ─────────────────────────────────────────────────────────────────────────────
test-search:   ## Run the 3 reference search queries (no extra args needed)
	@echo "\n\033[1;34m▶ Query 1 – semantic only\033[0m"
	uv run python -m qdrant_rag.rag.cli search \
	  "payment failed" \
	  --url $(QDRANT_URL) --top-k $(TOP_K) --candidates $(CANDIDATES)
	@echo "\n\033[1;34m▶ Query 2 – keyword filters: product + priority\033[0m"
	uv run python -m qdrant_rag.rag.cli search \
	  "crash on upload" \
	  --url $(QDRANT_URL) --top-k $(TOP_K) --candidates $(CANDIDATES) \
	  --product "Web Portal" --priority "Urgent"
	@echo "\n\033[1;34m▶ Query 3 – mixed filters: language + bool + date range\033[0m"
	uv run python -m qdrant_rag.rag.cli search \
	  "subscription issue" \
	  --url $(QDRANT_URL) --top-k $(TOP_K) --candidates $(CANDIDATES) \
	  --language "French" --sla-breached --created-from "2023-01-01"

# ─────────────────────────────────────────────────────────────────────────────
# Demo  (full pipeline from scratch)
#
#   make demo           → install + qdrant-up + create-collection + index-sample + test-search
#   make demo-full      → same but indexes the entire 200k CSV
# ─────────────────────────────────────────────────────────────────────────────
demo:          ## Full pipeline demo using the 500-row sample (default)
	@echo "\033[1;32m=== QDRANT RAG DEMO (sample) ===\033[0m"
	$(MAKE) install
	$(MAKE) qdrant-up
	$(MAKE) recreate-collection
	$(MAKE) index-sample
	$(MAKE) test-search

demo-full:     ## Full pipeline demo indexing the entire 200k CSV
	@echo "\033[1;32m=== QDRANT RAG DEMO (full dataset) ===\033[0m"
	$(MAKE) install
	$(MAKE) qdrant-up
	$(MAKE) recreate-collection
	$(MAKE) index
	$(MAKE) test-search

# ─────────────────────────────────────────────────────────────────────────────
# Collection info
# ─────────────────────────────────────────────────────────────────────────────
info:          ## Print collection statistics
	uv run python -m qdrant_rag.rag.cli info --url $(QDRANT_URL)

# ─────────────────────────────────────────────────────────────────────────────
# Code quality
# ─────────────────────────────────────────────────────────────────────────────
lint:          ## Run ruff linter
	uv run ruff check src/

fmt:           ## Format code with ruff
	uv run ruff format src/
