#!/bin/bash
# Sync wiki_content/*.md to the GitHub Wiki of this repository.
#
# Usage:
#   ./scripts/upload_wiki.sh                  # clone wiki to a temp dir, sync, push
#   WIKI_DIR=/path/to/clone ./scripts/upload_wiki.sh   # reuse an existing clone
#
# wiki_content/README.md is intentionally excluded — it documents the sync
# process itself and is not a wiki page.
set -euo pipefail

REPO_SLUG="${REPO_SLUG:-0x0000dead/whales-identification}"
WIKI_URL="${WIKI_URL:-https://github.com/${REPO_SLUG}.wiki.git}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${ROOT_DIR}/wiki_content"

if [[ ! -d "${SRC_DIR}" ]]; then
    echo "error: ${SRC_DIR} not found" >&2
    exit 1
fi

CLEANUP=0
if [[ -z "${WIKI_DIR:-}" ]]; then
    WIKI_DIR="$(mktemp -d)/wiki"
    CLEANUP=1
    echo "Cloning ${WIKI_URL} ..."
    git clone --depth 1 "${WIKI_URL}" "${WIKI_DIR}"
fi

for page in "${SRC_DIR}"/*.md; do
    name="$(basename "${page}")"
    [[ "${name}" == "README.md" ]] && continue
    cp "${page}" "${WIKI_DIR}/${name}"
done

cd "${WIKI_DIR}"
if git diff --quiet && [[ -z "$(git status --porcelain)" ]]; then
    echo "Wiki already up to date."
else
    git add -A
    git commit -m "docs: sync wiki from wiki_content/"
    git push
    echo "Wiki updated."
fi

if [[ "${CLEANUP}" == "1" ]]; then
    rm -rf "$(dirname "${WIKI_DIR}")"
fi
