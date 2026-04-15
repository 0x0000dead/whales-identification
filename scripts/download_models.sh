#!/bin/bash
# Download EcoMarineAI model checkpoints from Hugging Face Hub.
#
# Why the version pin: huggingface_hub 0.21+ removed the standalone
# `huggingface-cli` binary from the default install path on some platforms,
# which broke this script silently. Pinning to 0.20.3 keeps the CLI available
# regardless of what other packages pull in.
set -euo pipefail

HF_REPO="${HF_REPO:-baltsat/Whales-Identification}"
TARGET_DIR="${TARGET_DIR:-models}"
HF_HUB_VERSION="0.20.3"

mkdir -p "${TARGET_DIR}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "huggingface-cli not found; installing huggingface_hub==${HF_HUB_VERSION}..."
    python3 -m pip install --quiet "huggingface_hub==${HF_HUB_VERSION}"
fi

# Files to download. Each entry: "<filename>:<optional_alias>"
# Aliases let us rename on disk so model-e15.pt is what the service looks for.
MODELS=(
    "resnet101.pth"
    "vit_l32_best.pth:model-e15.pt"
)

for entry in "${MODELS[@]}"; do
    src="${entry%%:*}"
    dst="${entry##*:}"
    [ "${src}" = "${dst}" ] && dst="${src}"

    echo "Downloading ${src} from ${HF_REPO} → ${TARGET_DIR}/${dst}..."
    if huggingface-cli download "${HF_REPO}" "${src}" \
            --repo-type model \
            --local-dir "${TARGET_DIR}" \
            --local-dir-use-symlinks False \
            --quiet 2>/dev/null; then
        if [ "${src}" != "${dst}" ] && [ -f "${TARGET_DIR}/${src}" ]; then
            mv "${TARGET_DIR}/${src}" "${TARGET_DIR}/${dst}"
        fi
        echo "  ✓ ${TARGET_DIR}/${dst}"
    else
        echo "  ! ${src} not available in ${HF_REPO}; skipping."
    fi
done

echo "Model download complete. Files in ${TARGET_DIR}/:"
ls -lh "${TARGET_DIR}/" || true
