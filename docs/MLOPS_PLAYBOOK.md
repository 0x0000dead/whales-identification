# MLOps playbook

Operational runbook for the EcoMarineAI inference service. Read this when you're:

- Deploying a new model version.
- Investigating a drift alert.
- Rolling back a bad release.
- Answering "why did the service return weird predictions yesterday?".

---

## 1. Model registry

Source of truth: `models/registry.json`.

```json
{
  "schema_version": "1.0",
  "active": "effb4_15k",
  "models": [
    {
      "name": "effb4_15k",
      "display_name": "EfficientNet-B4 ArcFace 13 837-class",
      "version": "1.0.0",
      "architecture": "CetaceanIdentificationModel",
      "weights_url": "https://huggingface.co/0x0000dead/ecomarineai-cetacean-effb4/resolve/main/efficientnet_b4_512_fold0.ckpt",
      "sha256": null,
      "trained_at": "2022-04-18",
      "metrics_snapshot": "reports/metrics_baseline.json"
    },
    ...
  ]
}
```

### Promoting a new model

1. Train and save the checkpoint (`whales_identify/train.py` or an external notebook).
2. Upload to HuggingFace under a versioned filename: `efficientnet_b4_v1.1.0.ckpt`.
3. Run metrics locally: `python3 scripts/compute_metrics.py`. Confirm TNR ≥ 0.90, TPR ≥ 0.85.
4. Append a new entry to `models/registry.json` (increment `version`, update `weights_url`).
5. Update `scripts/download_models.sh` to match.
6. Commit, push, and watch `.github/workflows/metrics.yml` — it runs the regression gate vs `metrics_baseline.json`.
7. If CI is green, promote: update `"active"` in `registry.json` and merge.
8. If CI is red: the regression is real. Investigate before promoting.

### Demoting / rolling back

Simply change `"active"` back to the previous entry in `registry.json` and commit. No code change. Restart pods.

---

## 2. Drift monitoring

Two signals are exposed via `/metrics` and `/v1/drift-stats`:

| Signal                   | Where              | Alarm threshold                 |
|--------------------------|--------------------|---------------------------------|
| `cetacean_score_avg`     | `/metrics` gauge   | Drops > 0.10 pp from baseline   |
| `score_mean` (window)    | `/v1/drift-stats`  | Drops > 0.10 pp from baseline   |
| `alarms_total`           | `/v1/drift-stats`  | Non-zero = active drift         |
| `rejections_by_reason{reason="not_a_marine_mammal"}` rate | `/metrics` | > 50 % of requests = systemic issue |

### DriftMonitor internals

`whales_be_service/src/whales_be_service/monitoring/drift.py` keeps a 1 000-sample rolling deque of `cetacean_score` and `probability` values. On every prediction, `record()` appends and — if the window holds ≥ 50 samples AND a baseline is set — checks `(baseline - window_mean) ≥ alarm_drop`. A positive check logs `WARNING` and bumps `alarms_total`.

Baseline is not set automatically (to avoid silent drift acceptance). To set it, parse `metrics_baseline.json` and pass it at pipeline construction. See `whales_be_service/src/whales_be_service/inference/registry.py`.

### Responding to an alarm

1. Check `rejections_by_reason` — if `not_a_marine_mammal` is suddenly dominant, either:
   - the upstream image source has started sending non-cetacean input (user workflow regression), or
   - the gate threshold is too aggressive (calibration drift).
2. Run `python3 scripts/calibrate_clip_threshold.py` on the latest `data/test_split/` — does it pick a very different threshold?
3. If the threshold is the problem, update `configs/anti_fraud_threshold.yaml` and restart.
4. If the input distribution has shifted, collect recent samples and extend `data/test_split/` so the threshold can be recalibrated on realistic data.

---

## 3. Rollback procedures

### Fast rollback (image tag)

```bash
docker pull ghcr.io/0x0000dead/ecomarine-backend:v1.0.0
docker compose up -d backend
```

### Slow rollback (code + weights)

```bash
git revert <bad commit>
git push
# wait for CI + new image build
kubectl rollout restart deployment/ecomarine-backend
```

### Emergency: disable the anti-fraud gate

If the gate is misbehaving and rejecting everything, set `threshold: 0.0` in `configs/anti_fraud_threshold.yaml` and restart. This effectively accepts everything. Treat as a fire-drill measure — re-calibrate within 24 h.

---

## 4. On-call runbook (common incidents)

### Incident: `availability_percent < 95 %`

Check `errors_total` rate. Common causes:
- **OOM on 20+ MB images** — uvicorn worker killed by cgroup. Increase memory limits in the Deployment.
- **HF weights download failed on pod restart** — check `docker-entrypoint.sh` logs. Fallback: pre-populate `/app/src/whales_be_service/models/` via a PVC.
- **Rate limiter flood** — one IP is spamming. Check nginx / ingress logs; block the IP.

### Incident: `p95 latency > 2 s`

- Too many concurrent requests? Scale horizontally (`kubectl scale --replicas=4`).
- First request after pod restart always hits the cold-load path (~5 s). Expected.
- GPU suddenly not picked up? `torch.cuda.is_available()` may have returned False; check driver logs.

### Incident: user says "it called my whale a dolphin"

Remember: the identification model knows 30 species but only ~13 837 individuals. For unseen individuals the top-1 is unreliable. Tell the user to rely on `id_animal` (species) rather than `class_animal` (individual). The `cetacean_score` value is the honest "is this a cetacean at all" signal.

---

## 5. Deployment cadence

| Release type        | Frequency    | Approval                    |
|---------------------|--------------|-----------------------------|
| Patch (bugfix)      | On demand    | 1 reviewer, CI green        |
| Minor (new feature) | Bi-weekly    | 1 reviewer + metrics CI     |
| Major (model v2.0)  | Quarterly    | 2 reviewers + manual QA     |
| Hotfix              | Within 1 h   | 1 reviewer, skipping labs   |

---

## 6. Data lineage & reproducibility

Every prediction is reproducible given:

1. The input image (hash / bytes).
2. The model `version` (from `/metrics` or `model_version` field in the response).
3. The calibration `threshold` (from `configs/anti_fraud_threshold.yaml`).

We do **not** log input images by default (privacy + storage). If a reviewer needs to replay a specific case, they can:

- Re-upload the image via `/v1/predict-single` against the pinned version.
- Or compute the hash, diff against `metrics_baseline.json`, and replay the calibration numbers.

---

## 7. Secrets management

- **Kaggle API key** — lives in `~/.kaggle/kaggle.json` on developer machines and CI secrets on GitHub. Never committed.
- **HuggingFace token** — only needed when pushing to `0x0000dead/ecomarineai-cetacean-effb4`. Stored in GH secrets as `HF_TOKEN`. The service itself only reads from the repo (public), so no token is needed at runtime.
- **SQLite / PostgreSQL DSN** — passed via CLI flag or env var, never committed.
- **If a secret is accidentally committed**: rotate immediately, then follow the git object rewrite procedure (`git filter-repo`).

---

## 8. Observability dashboard

Recommended Grafana panels:

1. **Availability** — `availability_percent` over 7 days, threshold line at 95%.
2. **Throughput** — `rate(predictions_total[5m])` + `rate(rejections_total[5m])` stacked.
3. **Latency** — `latency_avg_ms` + histogram of request durations.
4. **Anti-fraud breakdown** — `rejections_by_reason` pie chart.
5. **Drift** — `cetacean_score_avg` with a baseline reference line.
6. **Errors** — `rate(errors_total[5m])` alert-backed.
