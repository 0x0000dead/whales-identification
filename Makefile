.PHONY: help install lint format test test-cov build up down logs run \
        download-models smoke compute-metrics calibrate-clip \
        pre-commit-install

## EcoMarineAI — developer commands (make help for usage)

help:
	@echo ""
	@echo "EcoMarineAI — available commands:"
	@echo ""
	@echo "  Setup"
	@echo "    make install          Install Python + JS dependencies"
	@echo "    make pre-commit-install  Wire pre-commit hooks"
	@echo "    make download-models  Download model weights from HuggingFace"
	@echo ""
	@echo "  Code quality"
	@echo "    make lint             Ruff lint + format check (CI-equivalent)"
	@echo "    make format           Auto-format with Ruff (modifies files)"
	@echo ""
	@echo "  Testing"
	@echo "    make test             Fast unit tests (no model weights required)"
	@echo "    make test-cov         Unit tests + HTML coverage report"
	@echo "    make smoke            End-to-end smoke test (requires running stack)"
	@echo ""
	@echo "  Run"
	@echo "    make up               Build + start full stack in Docker"
	@echo "                           Backend:  http://localhost:8000/docs"
	@echo "                           Frontend: http://localhost:8080"
	@echo "    make down             Stop all containers"
	@echo "    make logs             Tail container logs"
	@echo "    make run              Dev server (hot-reload, no Docker)"
	@echo ""
	@echo "  ML operations"
	@echo "    make compute-metrics  Rebuild metrics report from test_split"
	@echo "    make calibrate-clip   Re-calibrate CLIP threshold"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────

install:
	cd whales_be_service && poetry install
	cd frontend && npm install

pre-commit-install:
	cd whales_be_service && poetry run pre-commit install

download-models:
	bash scripts/download_models.sh

# ── Code quality ──────────────────────────────────────────────────────────────

lint:
	cd whales_be_service && poetry run ruff check . && poetry run ruff format --check .

format:
	cd whales_be_service && poetry run ruff format . && poetry run ruff check --fix .

# ── Testing ───────────────────────────────────────────────────────────────────

test:
	cd whales_be_service && poetry run pytest -m "not slow and not integration" -v

test-cov:
	cd whales_be_service && poetry run pytest -m "not slow and not integration" \
		--cov=src --cov-report=term --cov-report=html

smoke:
	bash scripts/smoke_test.sh

# ── Run ───────────────────────────────────────────────────────────────────────

build:
	docker compose build

up:
	docker compose up -d --build
	@echo ""
	@echo "  Backend:  http://localhost:8000/docs"
	@echo "  Frontend: http://localhost:8080"
	@echo ""

down:
	docker compose down

logs:
	docker compose logs -f

run:
	cd whales_be_service && poetry run python -m uvicorn whales_be_service.main:app \
		--reload --host 0.0.0.0 --port 8000

# ── ML operations ─────────────────────────────────────────────────────────────

compute-metrics:
	python3 scripts/compute_metrics.py \
		--manifest data/test_split/manifest.csv \
		--output-json reports/metrics_latest.json \
		--output-md reports/METRICS.md \
		--update-model-card

calibrate-clip:
	python3 scripts/calibrate_clip_threshold.py \
		--manifest data/test_split/manifest.csv
