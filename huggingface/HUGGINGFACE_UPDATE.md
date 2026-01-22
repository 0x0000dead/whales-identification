# Updating HuggingFace Repository

This directory contains files that should be synchronized to the HuggingFace repository [baltsat/Whales-Identification](https://huggingface.co/baltsat/Whales-Identification).

## Files

| File        | Purpose                                                       |
| ----------- | ------------------------------------------------------------- |
| `README.md` | Model card with YAML frontmatter (sets license to Apache 2.0) |
| `LICENSE`   | Apache 2.0 license text with usage restrictions               |

## Why Update?

The HuggingFace repository was initially created with MIT license, but the project uses **Apache 2.0 with Usage Restrictions** (due to training data constraints from HappyWhale and ImageNet). This inconsistency was identified in the expert review (NTO).

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
huggingface-cli upload baltsat/Whales-Identification \
    huggingface/README.md README.md \
    --repo-type model \
    --commit-message "Update model card with Apache 2.0 license"

# Upload LICENSE file
huggingface-cli upload baltsat/Whales-Identification \
    huggingface/LICENSE LICENSE \
    --repo-type model \
    --commit-message "Add Apache 2.0 license file"
```

### Option 3: Manual Upload via Web UI

1. Go to https://huggingface.co/baltsat/Whales-Identification
2. Click "Files and versions" tab
3. Click "Add file" → "Upload files"
4. Upload `README.md` and `LICENSE` from this directory
5. Commit with message: "Update model card with Apache 2.0 license"

## Verification

After updating, verify the license shows "Apache 2.0":

1. Visit https://huggingface.co/baltsat/Whales-Identification
2. Check the license badge in the repository header
3. It should show "apache-2.0" instead of "mit"

## YAML Frontmatter

The `README.md` contains YAML frontmatter that HuggingFace uses to set metadata:

```yaml
---
license: apache-2.0
license_name: Apache 2.0 with Usage Restrictions
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

The key field is `license: apache-2.0` which updates the repository license.

## Keeping in Sync

When updating model documentation or license terms:

1. Edit files in this `huggingface/` directory
2. Run `./scripts/update_huggingface.sh` to sync changes
3. Verify changes at https://huggingface.co/baltsat/Whales-Identification
