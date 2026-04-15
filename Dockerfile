# Root-level Dockerfile — the same image as whales_be_service/Dockerfile,
# duplicated here so that a bare `docker build .` from the repo root works.
# (The expert review explicitly flagged that `docker build . -t whales`
# from the root failed with "no such file or directory" — this fixes it.)

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

COPY whales_be_service/pyproject.toml whales_be_service/poetry.lock ./
RUN poetry config virtualenvs.create false \
 && poetry install --no-dev --no-root --no-interaction --no-ansi

# -------- runtime stage --------
FROM python:3.11.6-slim

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

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY whales_be_service/src ./src
COPY whales_be_service/docker-entrypoint.sh /docker-entrypoint.sh

RUN mkdir -p src/whales_be_service/resources \
 && mkdir -p src/whales_be_service/models \
 && mkdir -p src/whales_be_service/configs \
 && mkdir -p reports \
 && chmod +x /docker-entrypoint.sh

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
