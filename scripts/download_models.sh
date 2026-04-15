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
#
# Integrity: every downloaded file is verified against models/checksums.sha256.
# Missing or mismatched hashes fail the script. Transient network failures are
# retried 3× with exponential backoff (2s / 4s / 8s) before giving up.
set -euo pipefail

PRIMARY_REPO="${PRIMARY_REPO:-0x0000dead/ecomarineai-cetacean-effb4}"
LEGACY_REPO="${LEGACY_REPO:-baltsat/Whales-Identification}"
MODELS_DIR="${MODELS_DIR:-whales_be_service/src/whales_be_service/models}"
RESOURCES_DIR="${RESOURCES_DIR:-whales_be_service/src/whales_be_service/resources}"
CONFIGS_DIR="${CONFIGS_DIR:-whales_be_service/src/whales_be_service/configs}"
REPORTS_DIR="${REPORTS_DIR:-reports}"
CHECKSUMS_FILE="${CHECKSUMS_FILE:-models/checksums.sha256}"
HF_HUB_VERSION="0.20.3"
MAX_RETRIES=3
SKIP_CHECKSUMS="${SKIP_CHECKSUMS:-0}"

mkdir -p "${MODELS_DIR}" "${RESOURCES_DIR}" "${CONFIGS_DIR}" "${REPORTS_DIR}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "huggingface-cli not found; installing huggingface_hub==${HF_HUB_VERSION}..."
    python3 -m pip install --quiet "huggingface_hub==${HF_HUB_VERSION}"
fi

# Pick a SHA256 implementation that works on both macOS and Linux runners.
if command -v sha256sum >/dev/null 2>&1; then
    SHA256_BIN="sha256sum"
elif command -v shasum >/dev/null 2>&1; then
    SHA256_BIN="shasum -a 256"
else
    echo "ERROR: neither sha256sum nor shasum available" >&2
    exit 2
fi

compute_sha() {
    # Prints only the hex digest (strips the filename the tools append).
    ${SHA256_BIN} "$1" | awk '{print $1}'
}

expected_sha_for() {
    local rel_path="$1"
    [ -f "${CHECKSUMS_FILE}" ] || { echo ""; return; }
    awk -v p="${rel_path}" '$2 == p {print $1; exit}' "${CHECKSUMS_FILE}"
}

verify_sha() {
    local file="$1" rel_path="$2"
    if [ "${SKIP_CHECKSUMS}" = "1" ]; then
        echo "  (checksum skipped by SKIP_CHECKSUMS=1)"
        return 0
    fi
    local expected
    expected="$(expected_sha_for "${rel_path}")"
    if [ -z "${expected}" ]; then
        echo "  (no checksum registered for ${rel_path}; add to ${CHECKSUMS_FILE})"
        return 0
    fi
    local actual
    actual="$(compute_sha "${file}")"
    if [ "${actual}" != "${expected}" ]; then
        echo "  ✗ SHA256 mismatch for ${rel_path}" >&2
        echo "    expected: ${expected}" >&2
        echo "    actual:   ${actual}"   >&2
        return 1
    fi
    echo "  ✓ SHA256 OK"
}

fetch() {
    local repo="$1" file="$2" dest_dir="$3"
    echo "→ ${repo} / ${file} → ${dest_dir}/"

    local attempt=1 delay=2
    while [ ${attempt} -le ${MAX_RETRIES} ]; do
        if huggingface-cli download "${repo}" "${file}" \
            --repo-type model \
            --local-dir "${dest_dir}" \
            --local-dir-use-symlinks False \
            --quiet 2>/dev/null; then
            local rel_path="${dest_dir}/${file}"
            if verify_sha "${rel_path}" "${rel_path}"; then
                return 0
            fi
            # checksum mismatch: drop the bad file and retry
            rm -f "${rel_path}"
        fi

        if [ ${attempt} -lt ${MAX_RETRIES} ]; then
            echo "  retry ${attempt}/${MAX_RETRIES} in ${delay}s..."
            sleep ${delay}
            delay=$((delay * 2))
        fi
        attempt=$((attempt + 1))
    done

    echo "  (skip: ${file} not available in ${repo} after ${MAX_RETRIES} attempts)"
    return 0
}

echo "=== Primary: EfficientNet-B4 ArcFace (${PRIMARY_REPO}) ==="
fetch "${PRIMARY_REPO}" "efficientnet_b4_512_fold0.ckpt" "${MODELS_DIR}"
fetch "${PRIMARY_REPO}" "encoder_classes.npy"            "${MODELS_DIR}"
fetch "${PRIMARY_REPO}" "species_map.csv"                "${RESOURCES_DIR}"
fetch "${PRIMARY_REPO}" "anti_fraud_threshold.yaml"      "${CONFIGS_DIR}"
fetch "${PRIMARY_REPO}" "metrics_baseline.json"          "${REPORTS_DIR}"

echo "=== Legacy: baltsat/Whales-Identification ==="
fetch "${LEGACY_REPO}" "resnet101.pth" "${MODELS_DIR}"

echo "=== Contents of ${MODELS_DIR}/: ==="
ls -lh "${MODELS_DIR}/" || true
echo "=== Contents of ${RESOURCES_DIR}/: ==="
ls -lh "${RESOURCES_DIR}/" || true
echo "Model download complete."
