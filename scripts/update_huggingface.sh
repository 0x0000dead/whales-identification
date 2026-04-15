#!/bin/bash
# Update HuggingFace repository with model card and license.
#
# This script uploads the README.md (model card) and LICENSE files to
# HuggingFace, which will update the repository license metadata to
# **CC-BY-NC-4.0** (Creative Commons Attribution-NonCommercial 4.0 International).
#
# IMPORTANT: the license MUST remain cc-by-nc-4.0. Earlier drafts used
# Apache 2.0, which was flagged by Экспертиза 2.0 §1.1 as incompatible
# with the upstream Happy Whale dataset licence. Do NOT revert to Apache 2.0
# — the model weights inherit CC-BY-NC-4.0 from training data.
#
# Prerequisites:
#   1. Install huggingface_hub: pip install huggingface_hub==0.20.3
#   2. Login to HuggingFace: huggingface-cli login
#   3. Have write access to the primary repository.
#
# Usage:
#   ./scripts/update_huggingface.sh            # default: primary effb4 repo
#   HF_REPO=custom/repo ./scripts/update_huggingface.sh

set -euo pipefail

HF_REPO="${HF_REPO:-0x0000dead/ecomarineai-cetacean-effb4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
HF_DIR="$PROJECT_ROOT/huggingface"

echo "=== HuggingFace Repository Update Script ==="
echo "Repository: $HF_REPO"
echo "License target: CC-BY-NC-4.0"
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

# Sanity check: make sure the README doesn't accidentally contain apache-2.0
if grep -iq "apache.*2\.0\|apache license" "$HF_DIR/README.md"; then
    echo "Error: $HF_DIR/README.md still contains Apache 2.0 reference."
    echo "Please update the license to cc-by-nc-4.0 before uploading."
    echo "See huggingface/HUGGINGFACE_UPDATE.md for the required frontmatter."
    exit 2
fi

echo "Uploading README.md (model card) to $HF_REPO..."
huggingface-cli upload "$HF_REPO" "$HF_DIR/README.md" README.md \
    --repo-type model \
    --commit-message "Update model card with CC-BY-NC-4.0 license"

echo ""
echo "Uploading LICENSE to $HF_REPO..."
huggingface-cli upload "$HF_REPO" "$HF_DIR/LICENSE" LICENSE \
    --repo-type model \
    --commit-message "Add CC-BY-NC-4.0 license file"

echo ""
echo "=== Update Complete ==="
echo ""
echo "The HuggingFace repository license should now show 'cc-by-nc-4.0'"
echo "Verify at: https://huggingface.co/$HF_REPO"
echo ""
echo "Note: It may take a few minutes for the license badge to update."
