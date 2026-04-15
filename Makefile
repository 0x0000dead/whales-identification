.PHONY: install lint lint-fix test test-slow smoke compute-metrics calibrate-clip download-models download-test-set run docker-up docker-down

install:
	cd whales_be_service && poetry install

lint:
	cd whales_be_service && poetry run black --check . && poetry run flake8 .

lint-fix:
	cd whales_be_service && poetry run black . && poetry run isort .

test:
	cd whales_be_service && poetry run pytest -m "not slow" -v

test-slow:
	cd whales_be_service && poetry run pytest -m slow -v

download-models:
	bash scripts/download_models.sh

download-test-set:
	python3 scripts/download_test_set.py --target data/test_split

calibrate-clip:
	python3 scripts/calibrate_clip_threshold.py --manifest data/test_split/manifest.csv

compute-metrics:
	python3 scripts/compute_metrics.py \
		--manifest data/test_split/manifest.csv \
		--output-json reports/metrics_latest.json \
		--output-md reports/METRICS.md \
		--update-model-card

smoke:
	bash scripts/smoke_test.sh

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

run:
	cd whales_be_service && poetry run python -m uvicorn whales_be_service.main:app --reload --host 0.0.0.0 --port 8000
