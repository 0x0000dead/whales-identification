# EcoMarineAI test split

Small in-repo evaluation set used by `scripts/compute_metrics.py` and
`scripts/calibrate_clip_threshold.py`. The full evaluation set (5k+ images)
lives behind a DVC remote and is fetched by `scripts/download_test_set.py`.

## Layout

```
test_split/
├── manifest.csv          # one row per image (label, species, source, license)
├── positives/            # marine mammal photos (whales, dolphins)
└── negatives/            # NON-cetacean distractors (text, people, buildings, cats, landscapes)
```

## Manifest schema

| column        | description                                                                |
|---------------|----------------------------------------------------------------------------|
| relpath       | path relative to `data/test_split/`                                        |
| label         | `cetacean` or `non_cetacean`                                               |
| individual_id | unique whale/dolphin ID (only for positives)                               |
| species       | species name (only for positives)                                          |
| source        | data source identifier (`internal`, `coco_val`, etc.)                      |
| license       | source license string                                                      |
| split         | dataset split (`train`, `val`, `test`) — this directory is `test` only     |
|

## Sourcing notes

- **Positives**: pulled from in-repo sample whales (`data/*.jpg`), augmented
  with additional aerial cetacean photographs from the Happy Whale community
  dataset (CC-BY-NC-4.0). Required for measuring sensitivity (TPR ≥ 85%) and
  identification top-1 / top-5 accuracy.
- **Negatives**: a deliberate mix of distractors that the system should reject:
  text screenshots, people, buildings, household animals, landscapes without
  wildlife, fish (hard negative — cetaceans are mammals), and sharks (hard
  negative — visually similar). Used to measure specificity (TNR ≥ 90%) of
  the CLIP anti-fraud gate.

## How to populate this directory

The manifest above lists 5 positive + 5 negative example rows. Drop your
actual image files into `positives/` and `negatives/` matching the
`relpath` entries, or extend the manifest with additional rows.

For a richer evaluation, run:

```bash
python scripts/download_test_set.py --target data/test_split
```

This pulls the extended set (~500 images) from the project DVC remote.

## Reproducibility

After populating the directory, recompute metrics via:

```bash
make compute-metrics
```

This writes `reports/metrics_latest.json`, `reports/METRICS.md`, and updates
`MODEL_CARD.md` between the `<!-- metrics:start -->` / `<!-- metrics:end -->`
markers.
