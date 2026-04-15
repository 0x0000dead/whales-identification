# Compliance & licensing

EcoMarineAI touches three regulated areas: **data licenses**, **software licenses**, and **environmental / ethical use**. This doc consolidates everything so a legal / ethics reviewer can audit in one place.

---

## 1. Software license (source code)

- **LICENSE** — MIT, © 2024 EcoMarineAI contributors (see `LICENSE` in the repo root).
- Permissive: commercial, modification, distribution all allowed.
- No patent grant (MIT).

## 2. Training data license

The combined training corpus has two sources; the derivative model inherits the **most restrictive** terms of both.

### 2.1 Happy Whale dataset

- Source: https://happywhale.com / https://www.kaggle.com/competitions/happy-whale-and-dolphin
- Licence: **CC-BY-NC-4.0** (Creative Commons Attribution-NonCommercial 4.0 International).
- Attribution required (see [LICENSE_DATA.md](../LICENSE_DATA.md)).
- Non-commercial use only — commercial exploitation requires direct permission from Happy Whale (`support@happywhale.com`).

### 2.2 Ministry of Natural Resources RF dataset

- Source: Ministry of Natural Resources and Ecology of the Russian Federation.
- Licence: Government research-only. Not redistributable. Covered by the ФСИ grant agreement.
- Attribution format documented in `LICENSE_DATA.md`.

### 2.3 Combined effect

| Use case                       | Permitted? | Notes |
|--------------------------------|:----------:|-------|
| Academic research              | ✓          | Attribute both sources |
| Educational use                | ✓          | Accredited institutions |
| Non-profit conservation        | ✓          | — |
| Scientific publications        | ✓          | Cite both datasets |
| Government monitoring (RF)     | ✓          | With ФСИ approval |
| Commercial products            | ✗          | Blocked by CC-BY-NC and gov restrictions |
| Open-source tools (MIT)        | ✓          | For non-commercial downstream use |
| Startups / for-profit orgs     | ✗          | Unless they obtain commercial licence |

## 3. Pre-trained model licenses

| Component                 | Upstream                                    | Licence      |
|---------------------------|---------------------------------------------|--------------|
| OpenCLIP ViT-B/32         | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`    | Apache 2.0  |
| EfficientNet-B4 (ImageNet)| timm `efficientnet_b4` pre-trained weights  | Apache 2.0  |
| ArcFace head (fine-tuned) | `ktakita/happywhale-exp004-effb4-trainall` | MIT (Kaggle user content — CC0/MIT mixture) |
| ResNet-101 (legacy)       | `baltsat/Whales-Identification`             | MIT         |
| rembg background removal  | `danielgatis/rembg`                        | MIT          |

All permissive, but each component's attribution requirement is preserved in [LICENSE_MODELS.md](../LICENSE_MODELS.md).

## 4. Anthropic Hugging Face mirror

`0x0000dead/ecomarineai-cetacean-effb4` on HF carries the combined licence: **CC-BY-NC-4.0** (taking the strictest of the inputs). The HF model card lists all upstream sources in the `datasets` front-matter and in the `## Licensing` section.

## 5. Third-party dependency audit

See [LICENSES_ANALYSIS.md](../LICENSES_ANALYSIS.md) for the full 159-dependency license breakdown. Summary:

- **99.4 % permissive** (MIT / BSD / Apache 2.0 / MPL).
- **0.6 % LGPL** (e.g. libmysqlclient transitively through aiomysql) — acceptable.
- **0 % GPL / AGPL** — deliberately excluded.

## 6. GDPR / personal data

- The service does **not** log input images by default.
- The drift monitor stores only aggregate statistics (rolling window of floats), no images.
- The SQLite/Postgres sinks store filename + predictions — no raw pixel data.
- If you bolt on image storage (for active-learning workflows), make sure to inform users and get consent for their photos being retained.
- No PII in training data (the `individual_id` column refers to whales, not people).

## 7. Environmental impact

- Training: one-off ~5 GPU-hours for EffB4 fold 0 → ~5 kg CO2 equivalent at typical data-centre mix. Documented in `DOCS/RESEARCH_NOTES.md`.
- Inference: ~490 ms CPU per image. A typical 1 000-photo survey uses ~8 minutes of CPU wall time → negligible.
- We do **not** recommend running the service 24/7 on an always-on GPU. Use CPU autoscaling.

## 8. Dual-use / ethical considerations

- The system is designed for **conservation and research**, not commercial exploitation of marine resources.
- It could theoretically be repurposed for whaling operations targeting specific individuals. We mitigate this by:
  - Documenting prohibited uses in `LICENSE_MODELS.md` ("Applications that harm marine mammals", "Surveillance for hunting purposes").
  - Binding the model to the non-commercial CC-BY-NC-4.0 upstream.
  - Refusing to provide a "species → habitat → best hunting coordinates" integration.
- Researchers who suspect misuse should contact the team and, separately, the upstream data providers.

## 9. ГОСТ 7.32-2017 alignment

The source code alignment is complete. The accompanying research report (НТО) must still follow ГОСТ rules for its Russian-language deliverable. A non-exhaustive checklist of what the code side enables:

| ГОСТ requirement                                            | Source artifact                                    |
|-------------------------------------------------------------|----------------------------------------------------|
| Библиографические ссылки на методы                          | [RESEARCH_NOTES.md](RESEARCH_NOTES.md) §6         |
| Воспроизводимость результатов                               | `scripts/compute_metrics.py`, `scripts/benchmark_*` |
| Структурированное описание архитектуры                      | [ML_ARCHITECTURE.md](ML_ARCHITECTURE.md)          |
| Документация пользовательского интерфейса                   | [USER_GUIDE_BIOLOGIST.md](USER_GUIDE_BIOLOGIST.md) |
| Документация API                                            | [API_REFERENCE.md](API_REFERENCE.md)              |
| План тестирования                                           | [TESTING_STRATEGY.md](TESTING_STRATEGY.md)        |
| Развёртывание и эксплуатация                                | [DEPLOYMENT.md](DEPLOYMENT.md), [MLOPS_PLAYBOOK.md](MLOPS_PLAYBOOK.md) |

## 10. Contact for compliance questions

For data-license questions: the upstream providers (Happy Whale and the Ministry of Natural Resources RF). For code-license questions: open a GitHub issue tagged `legal`.
