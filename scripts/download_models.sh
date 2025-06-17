#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

HF_REPO="baltsat/Whales-Identification"
MODEL_FILE="resnet101.pth"
TARGET_DIR="models"

# Create target directory if it doesn't exist
mkdir -p "${TARGET_DIR}"

echo "Downloading ${MODEL_FILE} from ${HF_REPO} to ${TARGET_DIR}..."

huggingface-cli download "${HF_REPO}" "${MODEL_FILE}" \
    --repo-type model \
    --local-dir "${TARGET_DIR}" \
    --local-dir-use-symlinks False \
    --quiet

# Check if the model file was downloaded successfully
if [ -f "${TARGET_DIR}/${MODEL_FILE}" ]; then
    echo "${MODEL_FILE} downloaded successfully to ${TARGET_DIR}/${MODEL_FILE}"
else
    echo "Error: Failed to download ${MODEL_FILE}."
    exit 1
fi

# Optional: Add other models here if needed, for example:
# MODELS_TO_DOWNLOAD=(
# "EfficientNet.h5"
# "resnet54.pth"
# "swin_t_best.pth"
# "vit_b16_best.bin"
# "vit_l32_best.pth"
# )
# for model in "${MODELS_TO_DOWNLOAD[@]}"; do
#     echo "Downloading ${model} from ${HF_REPO} to ${TARGET_DIR}..."
#     huggingface-cli download "${HF_REPO}" "${model}" \
#         --repo-type model \
#         --local-dir "${TARGET_DIR}" \
#         --local-dir-use-symlinks False \
#         --quiet
#     if [ -f "${TARGET_DIR}/${model}" ]; then
#         echo "${model} downloaded successfully."
#     else
#         echo "Error: Failed to download ${model}."
#         # Decide if you want to exit 1 here or just warn
#     fi
# done

echo "Model download process complete."
