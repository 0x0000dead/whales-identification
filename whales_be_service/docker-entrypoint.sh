#!/usr/bin/env bash
# Docker entrypoint that fetches ML assets on first boot and then launches
# uvicorn. Idempotent: if weights are already present, skip the download.
set -euo pipefail

MODELS_DIR="/app/src/whales_be_service/models"
RES_DIR="/app/src/whales_be_service/resources"
CONFIGS_DIR="/app/src/whales_be_service/configs"
HF_REPO="${HF_REPO:-0x0000dead/ecomarineai-cetacean-effb4}"

need_download() {
    [ ! -f "${MODELS_DIR}/efficientnet_b4_512_fold0.ckpt" ] \
        || [ ! -f "${MODELS_DIR}/encoder_classes.npy" ] \
        || [ ! -f "${RES_DIR}/species_map.csv" ]
}

mkdir -p "${MODELS_DIR}" "${RES_DIR}" "${CONFIGS_DIR}"

if need_download; then
    echo "[entrypoint] Model assets missing — downloading from ${HF_REPO}..."

    python3 - <<'PY'
import os
from huggingface_hub import hf_hub_download

REPO = os.environ.get("HF_REPO", "0x0000dead/ecomarineai-cetacean-effb4")
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/src/whales_be_service/models")
RES_DIR = os.environ.get("RES_DIR", "/app/src/whales_be_service/resources")
CONFIGS_DIR = os.environ.get("CONFIGS_DIR", "/app/src/whales_be_service/configs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(CONFIGS_DIR, exist_ok=True)

files = [
    ("efficientnet_b4_512_fold0.ckpt", MODELS_DIR),
    ("encoder_classes.npy", MODELS_DIR),
    ("species_map.csv", RES_DIR),
    ("anti_fraud_threshold.yaml", CONFIGS_DIR),
    ("metrics_baseline.json", "/app/reports"),
]
os.makedirs("/app/reports", exist_ok=True)

for filename, dest in files:
    try:
        p = hf_hub_download(
            repo_id=REPO,
            filename=filename,
            local_dir=dest,
            local_dir_use_symlinks=False,
        )
        print(f"  OK {filename} -> {p}")
    except Exception as e:
        print(f"  SKIP {filename}: {e}")
PY
else
    echo "[entrypoint] Model assets already present — skipping download."
fi

echo "[entrypoint] Starting uvicorn..."
exec python -m uvicorn whales_be_service.main:app --host 0.0.0.0 --port "${PORT:-8000}"
