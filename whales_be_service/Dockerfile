# -------- build stage --------
FROM python:3.11.6-slim AS builder

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      libssl-dev \
      libffi-dev \
      python3-dev \
      curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN pip install --no-cache-dir poetry==1.8.5

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
 && poetry install --no-dev --no-root --no-interaction --no-ansi

# -------- runtime stage --------
FROM python:3.11.6-slim

# Runtime-only system dependencies (OpenCV, libGL for rembg, etc.)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender-dev \
      libgomp1 \
      curl \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY src ./src

# Resources directory holds database.csv and species_map.csv; configs holds
# calibrated anti-fraud thresholds. Created empty — docker-entrypoint.sh
# populates them from Hugging Face on first boot.
RUN mkdir -p src/whales_be_service/resources \
 && mkdir -p src/whales_be_service/models \
 && mkdir -p src/whales_be_service/configs \
 && mkdir -p reports

# Entrypoint that fetches model weights from HF on first boot if missing,
# then launches uvicorn. Makes `docker compose up` Just Work without any
# host-side `scripts/download_models.sh` dance.
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Create non-root user and grant write access to the dirs the entrypoint
# downloads into.
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser \
 && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    ALLOWED_ORIGINS="*" \
    HF_REPO="0x0000dead/ecomarineai-cetacean-effb4"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["/docker-entrypoint.sh"]
