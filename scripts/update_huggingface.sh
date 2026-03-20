#!/bin/bash
# Update HuggingFace repository with model card and license
#
# This script uploads the README.md (model card) and LICENSE files to
# HuggingFace, which will update the repository license metadata from MIT
# to Apache 2.0.
#
# Prerequisites:
#   1. Install huggingface_hub: pip install huggingface_hub==0.20.3
#   2. Login to HuggingFace: huggingface-cli login
#   3. Have write access to baltsat/Whales-Identification repository
#
# Usage:
#   ./scripts/update_huggingface.sh

set -euo pipefail

HF_REPO="baltsat/Whales-Identification"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
HF_DIR="$PROJECT_ROOT/huggingface"

echo "=== HuggingFace Repository Update Script ==="
echo "Repository: $HF_REPO"
echo ""

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli not found"
    echo "Please install: pip install huggingface_hub==0.20.3"
    exit 1
fi

# Check if logged in
if ! huggingface-cli whoami &> /dev/null; then
    echo "Error: Not logged in to HuggingFace"
    echo "Please run: huggingface-cli login"
    exit 1
fi

# Check if files exist
if [ ! -f "$HF_DIR/README.md" ]; then
    echo "Error: $HF_DIR/README.md not found"
    exit 1
fi

if [ ! -f "$HF_DIR/LICENSE" ]; then
    echo "Error: $HF_DIR/LICENSE not found"
    exit 1
fi

echo "Uploading README.md (model card) to $HF_REPO..."
huggingface-cli upload "$HF_REPO" "$HF_DIR/README.md" README.md \
    --repo-type model \
    --commit-message "Update model card with Apache 2.0 license"

echo ""
echo "Uploading LICENSE to $HF_REPO..."
huggingface-cli upload "$HF_REPO" "$HF_DIR/LICENSE" LICENSE \
    --repo-type model \
    --commit-message "Add Apache 2.0 license file"

echo ""
echo "=== Update Complete ==="
echo ""
echo "The HuggingFace repository license should now show 'Apache 2.0'"
echo "Verify at: https://huggingface.co/$HF_REPO"
echo ""
echo "Note: It may take a few minutes for the license badge to update."
