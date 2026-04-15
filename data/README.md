# `data/` directory

This directory holds dataset artefacts used by EcoMarineAI for training,
evaluation, and runtime database lookups.

## Contents

| Path                          | Purpose                                                              |
|-------------------------------|----------------------------------------------------------------------|
| `data.csv`                    | Mapping individual_id ↔ species (legacy filename, stable schema)     |
| `backfin_annotations.csv`     | Bounding box annotations for the dorsal fin detector                 |
| `datasets/data.csv`           | Same schema, used by training notebooks                              |
| `test_split/`                 | Small evaluation set committed in-repo (see `test_split/README.md`)  |

## Note on legacy filenames

`data.csv` and `backfin_annotations.csv` originate from the Happy Whale
community dataset (CC-BY-NC-4.0). Filenames and ID columns are
preserved unchanged for reproducibility — every commit and notebook references
them by these paths. Do not rename them without coordinated updates across
`whales_identify/`, `research/notebooks/`, and `whales_be_service/resources/`.

## Larger artefacts (not in git)

| Asset           | Where it lives             | How to fetch                            |
|-----------------|----------------------------|------------------------------------------|
| Model weights   | HuggingFace repo           | `make download-models`                   |
| Extended test set | DVC remote               | `python scripts/download_test_set.py`    |
| Training images | not bundled                | Provided separately by the data owner    |
