#!/usr/bin/env bash
# EcoMarineAI — unified startup script (Linux / macOS).
#
# Responsibilities:
#   1. Verify Docker daemon is reachable.
#   2. Download model weights if the local weights dir is empty.
#   3. Bring the stack up in detached mode (dev or prod compose).
#   4. Wait for /health to return 200 OK, then print service URLs.
#
# Idempotent: re-running is safe — Docker Compose will reconcile to the
# desired state without recreating healthy containers.
#
# Usage:
#   ./scripts/start.sh            # dev compose (docker-compose.yml)
#   ./scripts/start.sh prod       # production compose (docker-compose.prod.yml)
#   SKIP_DOWNLOAD=1 ./scripts/start.sh
#
# Exit codes:
#   0  — stack healthy
#   1  — Docker unavailable
#   2  — model download failed
#   3  — compose failed to come up
#   4  — /health never returned 200 within HEALTH_TIMEOUT
set -euo pipefail

MODE="${1:-dev}"
COMPOSE_FILE="docker-compose.yml"
if [ "${MODE}" = "prod" ]; then
    COMPOSE_FILE="docker-compose.prod.yml"
fi

HEALTH_URL="${HEALTH_URL:-http://localhost:8000/health}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-180}"  # seconds — model download can be slow on first boot
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

# Resolve the repo root regardless of where the user invokes us from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

log()  { printf "\033[1;34m[start]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn] \033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[err]  \033[0m %s\n" "$*" >&2; }

# 1. Docker sanity check ------------------------------------------------------
log "Checking Docker daemon..."
if ! command -v docker >/dev/null 2>&1; then
    err "docker CLI not found. Install Docker Desktop or the Docker Engine first."
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    err "Docker daemon is not reachable. Start Docker and retry."
    exit 1
fi

# docker compose v2 is expected. Fall back to legacy docker-compose if needed.
if docker compose version >/dev/null 2>&1; then
    COMPOSE=(docker compose -f "${COMPOSE_FILE}")
elif command -v docker-compose >/dev/null 2>&1; then
    warn "Using legacy docker-compose v1; upgrade to Docker Compose v2 recommended."
    COMPOSE=(docker-compose -f "${COMPOSE_FILE}")
else
    err "Neither 'docker compose' nor 'docker-compose' is available."
    exit 1
fi

# 2. Model weights ------------------------------------------------------------
MODELS_DIR="whales_be_service/src/whales_be_service/models"
if [ "${SKIP_DOWNLOAD}" = "1" ]; then
    log "SKIP_DOWNLOAD=1 — skipping model download step."
elif [ -f "${MODELS_DIR}/efficientnet_b4_512_fold0.ckpt" ]; then
    log "Model weights already present — skipping download."
else
    log "Downloading model weights via scripts/download_models.sh..."
    if ! bash scripts/download_models.sh; then
        err "Model download failed. Fix network / credentials and retry."
        exit 2
    fi
fi

# 3. Bring the stack up -------------------------------------------------------
log "Launching Docker Compose stack (${COMPOSE_FILE})..."
if ! "${COMPOSE[@]}" up -d --remove-orphans; then
    err "'docker compose up' failed. Check container logs with:"
    err "    ${COMPOSE[*]} logs"
    exit 3
fi

# 4. Wait for /health ---------------------------------------------------------
log "Waiting for ${HEALTH_URL} (timeout ${HEALTH_TIMEOUT}s)..."
deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
while true; do
    if curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then
        log "Backend is healthy."
        break
    fi
    if [ "$(date +%s)" -ge "${deadline}" ]; then
        err "Health check timed out after ${HEALTH_TIMEOUT}s."
        err "Inspect logs: ${COMPOSE[*]} logs backend"
        exit 4
    fi
    sleep 2
done

log "Stack is up:"
log "  backend  → http://localhost:8000  (docs: /docs, metrics: /metrics)"
log "  frontend → http://localhost:8080"
log "Stop the stack with: ${COMPOSE[*]} down"
