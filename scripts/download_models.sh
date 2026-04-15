#!/bin/bash
# Download EcoMarineAI model checkpoints from Hugging Face Hub.
#
# Primary source: 0x0000dead/ecomarineai-cetacean-effb4 — EfficientNet-B4
# ArcFace trained on 13 837 individual cetaceans, with encoder_classes.npy and
# species_map.csv bundled for inference.
#
# Fallback: baltsat/Whales-Identification — historical checkpoints (ResNet101,
# ViT-B16, Swin-T), kept for backwards compatibility.
#
# Why the version pin: huggingface_hub 0.21+ moved `huggingface-cli` into an
# optional extra, so pinning to 0.20.3 keeps the CLI available regardless of
# what other packages pull in.
set -euo pipefail

PRIMARY_REPO="${PRIMARY_REPO:-0x0000dead/ecomarineai-cetacean-effb4}"
LEGACY_REPO="${LEGACY_REPO:-baltsat/Whales-Identification}"
MODELS_DIR="${MODELS_DIR:-whales_be_service/src/whales_be_service/models}"
RESOURCES_DIR="${RESOURCES_DIR:-whales_be_service/src/whales_be_service/resources}"
HF_HUB_VERSION="0.20.3"

mkdir -p "${MODELS_DIR}" "${RESOURCES_DIR}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "huggingface-cli not found; installing huggingface_hub==${HF_HUB_VERSION}..."
    python3 -m pip install --quiet "huggingface_hub==${HF_HUB_VERSION}"
fi

fetch() {
    local repo="$1" file="$2" dest_dir="$3"
    echo "→ ${repo} / ${file} → ${dest_dir}/"
    huggingface-cli download "${repo}" "${file}" \
        --repo-type model \
        --local-dir "${dest_dir}" \
        --local-dir-use-symlinks False \
        --quiet 2>/dev/null || echo "  (skip: ${file} not available in ${repo})"
}

echo "=== Primary: EfficientNet-B4 ArcFace (${PRIMARY_REPO}) ==="
fetch "${PRIMARY_REPO}" "efficientnet_b4_512_fold0.ckpt" "${MODELS_DIR}"
fetch "${PRIMARY_REPO}" "encoder_classes.npy" "${MODELS_DIR}"
fetch "${PRIMARY_REPO}" "species_map.csv" "${RESOURCES_DIR}"
fetch "${PRIMARY_REPO}" "anti_fraud_threshold.yaml" "whales_be_service/src/whales_be_service/configs"
fetch "${PRIMARY_REPO}" "metrics_baseline.json" "reports"

echo "=== Legacy: baltsat/Whales-Identification ==="
fetch "${LEGACY_REPO}" "resnet101.pth" "${MODELS_DIR}"

echo "=== Contents of ${MODELS_DIR}/: ==="
ls -lh "${MODELS_DIR}/" || true
echo "=== Contents of ${RESOURCES_DIR}/: ==="
ls -lh "${RESOURCES_DIR}/" || true
echo "Model download complete."
