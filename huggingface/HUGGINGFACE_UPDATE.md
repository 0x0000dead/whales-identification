# Updating HuggingFace Repository

This directory contains files that should be synchronized to the HuggingFace repositories
[baltsat/Whales-Identification](https://huggingface.co/baltsat/Whales-Identification) (legacy)
and [0x0000dead/ecomarineai-cetacean-effb4](https://huggingface.co/0x0000dead/ecomarineai-cetacean-effb4) (primary).

## Files

| File        | Purpose                                                        |
| ----------- | -------------------------------------------------------------- |
| `README.md` | Model card with YAML frontmatter (sets licence to CC-BY-NC-4.0) |
| `LICENSE`   | CC-BY-NC-4.0 licence text with usage restrictions              |

## Why Update?

Earlier drafts of this repo labelled the models as Apache 2.0. That was
**inconsistent** with the upstream Happy Whale dataset (CC-BY-NC-4.0), and
the expert review of the intermediate НТО (round 4, 19.01.2026) flagged the
mismatch. The correct canonical licence is **CC-BY-NC-4.0**, inherited from
the training data. Both this directory and both HuggingFace mirrors now
match that choice.

## How to Update

### Option 1: Using the Script (Recommended)

```bash
# Prerequisites
pip install huggingface_hub==0.20.3
huggingface-cli login

# Run the update script
./scripts/update_huggingface.sh
```

### Option 2: Manual Upload via CLI

```bash
# Login to HuggingFace
huggingface-cli login

# Upload README.md (this sets the license metadata)
huggingface-cli upload 0x0000dead/ecomarineai-cetacean-effb4 \
    huggingface/README.md README.md \
    --repo-type model \
    --commit-message "Update model card with CC-BY-NC-4.0 license"

# Upload LICENSE file
huggingface-cli upload 0x0000dead/ecomarineai-cetacean-effb4 \
    huggingface/LICENSE LICENSE \
    --repo-type model \
    --commit-message "Add CC-BY-NC-4.0 license file"
```

### Option 3: Manual Upload via Web UI

1. Go to https://huggingface.co/0x0000dead/ecomarineai-cetacean-effb4
2. Click "Files and versions" tab
3. Click "Add file" → "Upload files"
4. Upload `README.md` and `LICENSE` from this directory
5. Commit with message: "Update model card with CC-BY-NC-4.0 license"

## Verification

After updating, verify the license shows "cc-by-nc-4.0":

1. Visit https://huggingface.co/0x0000dead/ecomarineai-cetacean-effb4
2. Check the license badge in the repository header
3. It should show "cc-by-nc-4.0"

## YAML Frontmatter

The `README.md` contains YAML frontmatter that HuggingFace uses to set metadata:

```yaml
---
license: cc-by-nc-4.0
license_name: Creative Commons Attribution-NonCommercial 4.0 International
license_link: LICENSE
library_name: pytorch
pipeline_tag: image-classification
tags:
  - pytorch
  - vision
  - whale-identification
  - marine-mammals
  - conservation
---
```

The key field is `license: cc-by-nc-4.0` which updates the repository license. **Do not change this to `apache-2.0`** — Экспертиза 2.0 §1.1 explicitly prohibited Apache 2.0 for the model weights since they are derived from CC-BY-NC-4.0 Happy Whale data.

## Keeping in Sync

When updating model documentation or license terms:

1. Edit files in this `huggingface/` directory
2. Run `./scripts/update_huggingface.sh` to sync changes
3. Verify changes at https://huggingface.co/baltsat/Whales-Identification
