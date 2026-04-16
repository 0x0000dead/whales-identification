# FAQ

Short answers to the questions we get most often. Longer answers link to the relevant docs.

---

## General

### What is EcoMarineAI?

A free, open-source AI library and web service for identifying individual whales and dolphins from aerial photographs. See [SOLUTION_OVERVIEW.md](SOLUTION_OVERVIEW.md).

### Is it production-ready?

For **research and conservation work** — yes. Anti-fraud gate hits TNR = 90.2%, sensitivity = 95.0%, linear scalability (R² = 1.000), real metrics, full Docker deployment. For **safety-critical decisions affecting endangered-species populations** — use it as input to a biologist-reviewed workflow, not as a final oracle. See "Honest limitations" in [RESEARCH_NOTES.md](RESEARCH_NOTES.md) §7.

### Is it free?

Source code is MIT — yes, free forever. Model weights inherit CC-BY-NC-4.0 from the Happy Whale training data, which means **non-commercial** use only. Commercial users must obtain a separate licence from Happy Whale. See [COMPLIANCE.md](COMPLIANCE.md) §2.

### How do I cite it?

```bibtex
@software{ecomarineai2025,
  title  = {EcoMarineAI: Open Library for Cetacean Identification from Aerial Imagery},
  author = {Baltsat, K.I. and Tarasov, A.A. and Vandanov, S.A. and Serov, A.I.},
  year   = {2025},
  url    = {https://github.com/0x0000dead/whales-identification}
}
```

Also cite the Happy Whale dataset (CC-BY-NC-4.0) and, if relevant, the upstream checkpoint author (`ktakita/happywhale-exp004-effb4-trainall`).

---

## Installation

### How do I install it?

`git clone` + `docker compose up --build`. See [README.md](../README.md#-quickstart-для-бабушки-5-минут-даже-без-python-опыта) or [DEPLOYMENT.md](DEPLOYMENT.md).

### Do I need a GPU?

No. CPU-only inference gives p95 latency ≈ 540 ms, which is 12× under the TZ budget. A GPU brings it down to ~25 ms if you need higher throughput.

### How big is the download?

- Docker image: ~2.3 GB.
- Model weights: ~400 MB (EffB4 + ResNet101, cached in `models/`).
- Test split (already in the repo): 1.3 MB.

### Windows / macOS / Linux?

Yes to all three. Docker Desktop works on all; for native Linux you can `poetry install` directly.

### Why did `docker compose up` fail on first boot?

Most common causes:

1. **Docker Desktop not running** → start it first.
2. **Port 8080 or 8000 already in use** → `docker compose down`, change the mapping in `docker-compose.yml`.
3. **Out of disk** → the image + weights need ~3 GB free.
4. **Proxy / firewall blocking HuggingFace** → set `HTTPS_PROXY` or pre-populate `models/` manually via `scripts/download_models.sh`.

---

## Using the model

### How many species can it identify?

30 cetacean species. Full list in [MODEL_CARD.md](../MODEL_CARD.md). The species with the most training data are humpback whale, bottlenose dolphin, and blue whale.

### How many individual animals?

13 837 unique individuals. These are the animals the model saw during training. It can recognise them again in new photos with top-1 accuracy that varies by species (humpbacks: high; minkes: lower due to fewer training examples).

### What if my whale isn't in the training set?

The model still predicts a species (because species-level features generalise) but the `individual_id` will point to the closest known match. Trust `id_animal` more than `class_animal` in that case, and look at `probability` — low values (<0.1) signal "unseen individual".

### What happens if I upload something that isn't a whale?

The CLIP anti-fraud gate rejects it with `rejected: true, rejection_reason: "not_a_marine_mammal"`. Returns HTTP 200 (a rejection is a valid classification outcome). See [API_REFERENCE.md](API_REFERENCE.md).

### The gate rejected a photo that IS a whale — what now?

Look at `cetacean_score`. If it's between 0.2 and 0.5, the image is borderline (heavy crop, low light, partial animal). Try:

1. A tighter crop that foregrounds the animal.
2. Better lighting / less backlighting.
3. A clearer dorsal fin or fluke shot (the model was trained mostly on those angles).

If `cetacean_score` is very high but the identification confidence is low, rely on the species name (`id_animal`) rather than the individual ID.

### Can I make the gate less strict?

Yes — lower the threshold in `whales_be_service/src/whales_be_service/configs/anti_fraud_threshold.yaml`. For example, `0.2` accepts more borderline photos at the cost of specificity. Re-run `scripts/calibrate_clip_threshold.py` after adding more real examples to `data/test_split/`.

### Can I add a new species?

1. Collect labelled photos of the new species.
2. Add them to the training CSV with new `individual_id` values.
3. Re-fit the ArcFace head (`whales_identify/train.py`) — no backbone retraining needed if the new species is visually close to existing ones.
4. Push the new checkpoint to HuggingFace, bump the version in `models/registry.json`.

See [ROADMAP.md](ROADMAP.md) for the long-term extensibility plan.

---

## Data

### Where does the training data come from?

The **publicly verifiable** portion is **51 034 images × 15 587 individuals** from the Happy Whale Kaggle competition. The ТЗ also cites an additional private subset from the Ministry of Natural Resources RF (~29 k images). The currently-deployed EfficientNet-B4 checkpoint was trained on the public Happy Whale set only (fold 0) — see [MODEL_CARD.md](../MODEL_CARD.md) §Training Data for the full breakdown. The ТЗ 80 k aggregate refers to the *combined* Happy Whale + Ministry RF corpus.

### Can I download the training data?

Yes, the public part from Kaggle: https://www.kaggle.com/competitions/happy-whale-and-dolphin/data — requires a Kaggle account and accepting the competition rules. The in-repo `scripts/populate_test_split.py` reproduces a 100-positive subset for evaluation.

### What about the Ministry of Natural Resources data?

The Ministry RF dataset is covered by the ФСИ grant agreement and **cannot be redistributed** (terms in [LICENSE_DATA.md](../LICENSE_DATA.md) §2). As a practical consequence:
- Reproducibility of the ТЗ 80 k aggregate is limited to Happy Whale's 51 k public images.
- The deployed checkpoint does **not** currently use Ministry RF data — it was trained on the public Happy Whale set only. Retraining with the Ministry portion is on the roadmap once the grant's data-sharing protocol is finalised.
- This is an **honest limitation** we surface in the docs rather than hide behind marketing language.

### Can I use my own dataset?

Absolutely — the pipeline is data-agnostic. Train a new ArcFace head on your data, upload weights to HuggingFace, set `HF_REPO=yourorg/yourweights` in the Docker environment, and the service picks up your model on next restart.

---

## Integration

### How do I store predictions in a database?

Use `integrations/sqlite_sink.py` for SQLite or `integrations/postgres_sink.py` for PostgreSQL. See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md).

### Can I call the API from JavaScript / R / Julia?

Yes. The API is plain HTTP + multipart form data. Examples for curl, Python, JavaScript, and R are in [API_REFERENCE.md](API_REFERENCE.md) and [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md).

### How do I monitor the service in production?

Scrape `/metrics` with Prometheus (it's already Prometheus-formatted), visualize in Grafana. Recommended panels in [DEPLOYMENT.md](DEPLOYMENT.md) §"Monitoring stack" and [MLOPS_PLAYBOOK.md](MLOPS_PLAYBOOK.md) §8.

### Does it integrate with GBIF / OBIS / iNaturalist?

Not directly yet — but the SQLite schema is Darwin Core-friendly. A `darwin_core_sink.py` is planned for Q3 2026 (see [ROADMAP.md](ROADMAP.md)).

---

## Performance

### How fast is it?

Per-image p95 latency of 540 ms on a single CPU worker. With 4 workers on a 4-core VM you get ~7 images/second. With a GPU (T4 class) it's ~40 images/second per worker. See [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md) §3 and §8.

### Can I batch many images in one request?

Yes — ZIP them and POST to `/v1/predict-batch`. The whole batch counts as one rate-limited request.

### Does it scale linearly?

Yes. `scripts/benchmark_scalability.py` sweeps 10/25/50/100 images and fits a linear regression with **R² = 1.000**. Slope ≈ 482 ms/image.

### How much memory does it use?

~1.5 GB RSS after warmup (CLIP ~720 MB + EffB4 ~540 MB + overhead). See [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md) §7.

---

## Troubleshooting

### `ModuleNotFoundError: open_clip_torch`

The anti-fraud gate falls back to permissive mode (lets everything through) with an ERROR log. Install with `pip install open-clip-torch` — already listed in `whales_be_service/pyproject.toml`.

### `rembg` crashes on Python 3.14

Known upstream bug (rembg calls `sys.exit(1)` during import on 3.14). The service catches this and returns `mask: null`. Either downgrade to Python 3.11 or just ignore masks.

### The first request is really slow

That's the cold-load path — CLIP + EffB4 load on demand the first time. Subsequent requests are fast. Warmup happens inside the FastAPI `lifespan()`, so in Docker the first user request after container start is fast.

### CI metrics gate keeps failing

Either:

1. Your change really did regress the metrics — investigate with `scripts/compute_metrics.py`.
2. The test split changed — regenerate the baseline via `cp reports/metrics_latest.json reports/metrics_baseline.json`, commit, and retry.

---

## Contributing

### How do I report a bug?

Open an issue at https://github.com/0x0000dead/whales-identification/issues with:
- The command you ran.
- The expected vs actual behaviour.
- `docker compose logs` output if relevant.

### How do I submit a pull request?

Fork the repo, create a branch, run `pytest -m "not slow"` locally, open a PR against `main`. CI runs lint/unit/security/docker-build. See `wiki_content/Contributing.md`.

### Can I use this for my master's / PhD thesis?

Yes — MIT licence on code, CC-BY-NC-4.0 on data/models for non-commercial academic use. Please cite the project (see "How do I cite it?" above).
