# Critic Audit #1 — 2026-04-15

Reviewer: critic agent
Scope: full repository audit against `TASK/Техническое задание.txt`, `PLAN_STAGE3.md`, `work/stage3/extracted_docs/Экспертиза 2.0 2 этап.md`, `work/stage3/extracted_docs/Экспертизы, замечания ФСИ.md`, `work/stage3/extracted_docs/Отчет промежуточный 2 этап ИСПРАВЛЕННЫЙ.md`.
Mode: read-only.

---

## Executive summary

The 2-этап report has been textually corrected (licenses, wiki, huggingface_hub pin, ГОСТ), but **several showstoppers remain before Stage 3 can be сдан ФСИ без доработки**:

1. **Ложное заявление о Precision ≥ 80 %** по ТЗ §Параметр 1. «Precision 0.9048 / 0.9355» в README/MODEL_CARD/GRANT_DELIVERABLES/PERFORMANCE_REPORT/SOLUTION_OVERVIEW — это **precision бинарного CLIP анти-фрод гейта**, а не «Precision идентификации **особей**» (multiclass). Реальная метрика идентификации в `reports/metrics_latest.json` — **top-1 accuracy 0.22** (22 %), то есть ниже 80 % в ~3.6 раза. Тот же самый `compute_metrics.py` честно её считает, но документация подменяет её анти-фрод precision. Эксперт ФСИ это поймает сразу.
2. **Несколько PLAN_STAGE3.md деливераблов 3.1–3.8 отсутствуют целиком**: `docs_fsie/` (final НТО + 4 руководства), `k8s/deployment.yaml+hpa.yaml+ingress.yaml` (§3.2), `reports/LOAD_TEST.md`, `docs/QUICKSTART_COLAB.ipynb` (§3.3), `integrations/happywhale_sink`/`gbif_sink.py`/`inat_sink.py` (§3.4), `research/notebooks/10_hyperparameter_search.ipynb`+`reports/OPTIMIZATION.md` (§3.5), `research/notebooks/11_ensemble_architecture.ipynb`+`reports/ENSEMBLE.md` (§3.6), `docker-compose.prod.yml`+`scripts/start.sh`+`start.bat` (§3.7), webhook/export endpoints (§3.8).
3. **Остаточные замечания Экспертизы 2.0 не исправлены**: notebook 07 по-прежнему ссылается на `model-e15.pt`; notebook 12 содержит абсолютные пути `/Users/savandanov/...`; `wiki_content/Model-Cards.md`, `Usage.md`, `FAQ.md`, `Architecture.md`, `Testing.md` содержат ~15 упоминаний устаревшей `model-e15.pt`; `wiki_content/Contributing.md:748` и `huggingface/README.md:84` всё ещё маркируют модели как **Apache 2.0** (прямо запрещено Экспертизой 2.0 §1.1); `models_config.yaml::vit_l32.checkpoint` указывает на `models/model-e15.pt`, которого нет в HF; `models/registry.json::active == vit_l32` с `weights_path: models/model-e15.pt` — вразрез с `models_config.yaml::active_model == effb4_arcface`; код пайплайна возвращает `model_version: vit_l32-v1` (hard-coded в 4 файлах), хотя реально грузится `effb4-arcface-v1`.
4. **Фабрикованные числа в wiki и README**: `wiki_content/Model-Cards.md:88` даёт «Precision@1 93.2 %», таблица по 10 видам с sample counts «Humpback Whale 12543» — ни одна цифра не воспроизводится ни одним скриптом; `README.md:387-398` содержит таблицу сравнения 7 моделей с Precision 82–93 %, Availability 90–95 %, F1 0.79–0.92 — это те самые выдуманные числа, за которые уже дважды получены замечания ФСИ (round 3 КП 1: «random.uniform(0.85, 0.95)»).
5. **Противоречивые числа в docs**: `DOCS/GRANT_DELIVERABLES.md:13` — Precision **93.55 %**, `n=60`; `DOCS/ML_ARCHITECTURE.md:127` — Precision **0.9355**, top-1 **5/30**; `DOCS/PERFORMANCE_REPORT.md:26` — Precision **0.9048**, top-1 **22/100**; `reports/METRICS.md` — Precision **0.9048**, top-1 **0.22**; `DOCS/SOLUTION_OVERVIEW.md:52` — Precision **0.9048**. Трёх разных цифр в 5 документах быть не должно.

Riks level: **не сдать без доработки** — блокёров 5, HIGH — 11, MEDIUM — 9, LOW/nits — 6.

---

## Findings

### 🔴 Critical (блокируют сдачу)

#### [F1] Precision ≥ 80 % по ТЗ §Параметр 1 **не измерена** — подменена анти-фрод precision
- **Where:**
  - `reports/metrics_latest.json:15` → `anti_fraud.precision = 0.9048`, `identification.top1_accuracy = 0.22`
  - `reports/METRICS.md:18,27`
  - `MODEL_CARD.md:53,63`
  - `DOCS/GRANT_DELIVERABLES.md:13` — «**93.55 %**»
  - `DOCS/ML_ARCHITECTURE.md:127,136` — «Precision (PPV) 0.9355, Top-1 5/30»
  - `DOCS/PERFORMANCE_REPORT.md:26,38,164` — «Precision 90.48 %, Top-1 22/100; строка в Summary: `Precision … 90.48 % + Laplacian check`»
  - `DOCS/SOLUTION_OVERVIEW.md:52`
  - `README.md:19`
  - `scripts/compute_metrics.py:253-288` (считает precision бинарного гейта, а идентификационный top-1 остаётся 0.22)
- **Claimed:** Параметр 1 ТЗ (Precision идентификации ≥ 80 % на чётких 1920×1080, где чёткость — вариация Лапласа ≤ 5 % ниже среднего) — **выполнено**, Precision 88 % / 90.48 % / 93.55 %.
- **Actual:**
  1. Все цифры 88/90.48/93.55 % относятся к CLIP-антифрод гейту «is it a cetacean?», бинарная задача.
  2. ТЗ §Параметр 1 формулирует Precision как «Процент правильно идентифицированных **особей** морских млекопитающих от общего числа изображений, классифицированных как содержащие морских млекопитающих». Единственный в проекте прокси для «особей» — `identification.top1_accuracy` в `reports/metrics_latest.json`, который = **0.22** (22 %).
  3. Эксперт ФСИ Экспертизы 2.0 § «Требования ТЗ 1.1» явно требовал обеспечить возможность запуска и проверки Precision идентификации — при запуске `scripts/compute_metrics.py` эксперт увидит top-1 = 0.22, что не подтверждает заявленные 80 %+.
  4. `DOCS/PERFORMANCE_REPORT.md:41` честно признаёт: «Top-1 looks modest because the test split mixes all 5 Happy Whale k-folds while the public EfficientNet-B4 checkpoint was trained on fold 0 only». То есть команда знает, но не переделала evaluation ни фильтрацию на fold 0.
  5. Laplacian-gate из §Параметр 1 реализован (`scripts/compute_metrics.py:_laplacian_variance`), но top-1 **не пересчитывается** только на «чётких» изображениях (clarity >= threshold). Результат: ТЗ condition не применено.
- **Source:** TЗ §Параметр 1, Экспертиза 2.0 §«Требования ТЗ 1.1», PLAN_STAGE3.md «Статус параметра 1: ✅ 88 %+», `reports/metrics_latest.json`
- **Fix:**
  1. Пересчитать evaluation на подмножестве test split, пересекающемся с fold 0 тренировочного сета — или заново натренировать effb4 на всех 5 fold-ах.
  2. Ввести отдельную метрику `identification_precision` = правильных топ-1 предсказаний среди «чётких» изображений / все чёткие изображения в test split, и убедиться что она ≥ 0.80. Пока так — поставить явное «⚠️ параметр 1 под риском».
  3. Исправить MODEL_CARD, GRANT_DELIVERABLES, PERFORMANCE_REPORT, SOLUTION_OVERVIEW, README — чётко разделить «Anti-fraud binary precision» и «Individual identification precision». Не маркировать анти-фрод precision как удовлетворение ТЗ §Параметр 1.
  4. Либо переопределить §Параметр 1 как species-level precision (не individual-level) по согласованию с ФСИ и пересчитать по species_map.csv. `compute_metrics.py` не имеет такой метрики — надо добавить. Важно: это **пересмотр метрики в сторону смягчения**, что требует отдельного письма в ФСИ.

---

#### [F2] ТЗ §Параметр 12 «80 000 изображений, 1 000 особей» — не доказано для checkpoint, только декларативно
- **Where:**
  - `MODEL_CARD.md:28-29`
  - `DOCS/GRANT_DELIVERABLES.md:24`
  - `PLAN_STAGE3.md:234` — «✅ ~65k изображений, 13837 особей»
  - `reports/metrics_latest.json` (test split 202 image)
- **Claimed:** Параметр 12 выполнен: 15 587 индивидов × «на порядок выше 1 000». 80 k upstream corpus.
- **Actual:**
  1. Чекпоинт `efficientnet_b4_512_fold0.ckpt` натренирован на **51 034** Happy Whale images, fold 0 only (см. `MODEL_CARD.md:27`). То есть тренировочных изображений для **этого** чекпоинта ≈ 10 k (fold 0 ≈ 51034 / 5). Не 80 k.
  2. Оставшиеся 29 k до 80 k → «Ministry RF ≈ 29 k (private, ФСИ-covered, not redistributable)» — но **это не подтверждено**. Тренировка на Ministry RF данных не проводилась (checkpoint взят от `ktakita/happywhale-exp004-effb4-trainall`, upstream обучение Kaggle-only).
  3. Параметр 12 говорит про датасет обучения И тестирования; test split в репозитории — 100 позитивов + 102 негатива (202 image), что ≈ 0.25 % от 80 k.
  4. Это 4-й раз, когда экспертиза задаёт вопрос (round 1 КП 12, round 2 КП 11, round 3 КП 5, round 4 КП 1.1.1.3). Тема «80 000» → «рекомендуется, после согласования с Заказчиком, перевести работу в «Сбор данных» без агрегации». Без явного согласия ФСИ это остаётся замечанием.
- **Source:** ТЗ §Параметр 12, Экспертиза ФСИ round 3 КП 5
- **Fix:**
  1. Честно описать в Заключительном НТО: public Happy Whale ≈ 51 k, Ministry RF — не использовалась в данном чекпоинте (либо использовалась — тогда приложить audit trail).
  2. Если Ministry RF действительно передавалась команде по ФСИ-договору, но не использовалась в обучении — явно это зафиксировать в справке об устранении (чтобы ФСИ видел, что команда не скрыла).
  3. Предоставить тестовую выборку ≥ 5000 изображений (сейчас 202) для более осмысленной оценки Параметров 1, 8, 9, 11. 202 image даёт ±7 п.п. доверительный интервал на TNR/TPR — слишком широко для 80 %/85 %/90 %/0.60 порогов.

---

#### [F3] ТЗ §Параметр 7 Availability ≥ 95 % / 7 дней — **нет production-деплоя, нет 7-дневного измерения**
- **Where:**
  - `whales_be_service/src/whales_be_service/main.py:170-177` — `availability_percent` gauge
  - `DOCS/PERFORMANCE_REPORT.md:106-113`
  - `DOCS/GRANT_DELIVERABLES.md:19` — «100 % on smoke test»
  - `PLAN_STAGE3.md:229` — «⚠️ Нет production деплоя»
- **Claimed:** ✅ Availability endpoint exposes real metric, target met.
- **Actual:**
  1. `availability_percent` вычисляется из счётчиков `requests_total` и `errors_total` **в памяти процесса**. При рестарте обнуляется. 7-дневный мониторинг невозможен без внешнего TSDB (Prometheus + Grafana).
  2. Smoke test — <1 мин, не 7 дней. Заявление «100 % availability for smoke» не соответствует формулировке ТЗ («в периоде 7 дней»).
  3. Нет production-деплоя, нет https, нет внешнего URL, нет uptime check сервиса (типа UptimeRobot или k8s Prometheus). PLAN 3.2 требовал k8s deployment — не сделан.
- **Source:** ТЗ §Параметр 7, PLAN_STAGE3.md §3.2
- **Fix:**
  1. Поднять сервис на public URL (Render.com, Fly.io, Koyeb, free tier) минимум на 7 дней, настроить любой сторонний uptime monitor (Better Stack / UptimeRobot free).
  2. Приложить скриншот/отчёт с цифрами Availability за 7 дней в Заключительный НТО.
  3. Либо реализовать k8s deployment по PLAN 3.2 и поднять на реальном кластере. Без этого Параметр 7 **не выполнен**.

---

#### [F4] ТЗ §Параметр 6 «≥ 2 БД + ≥ 2 платформы мониторинга» — **≥ 2 БД ok, ≥ 2 платформы мониторинга спорно**
- **Where:**
  - `integrations/postgres_sink.py`, `integrations/sqlite_sink.py` — 2 БД ✓
  - `integrations/otel_sink.py` — OpenTelemetry ✓
  - `/metrics` Prometheus ✓
  - `DOCS/INTEGRATION_GUIDE.md:186-201` — «GBIF/OBIS/iNaturalist: A `darwin_core_sink.py` is on the roadmap for Q3 2026»
  - `DOCS/GRANT_DELIVERABLES.md:18` — «4 sinks: CSV, SQLite, PostgreSQL, HF Hub»
  - PLAN_STAGE3.md:148-152 — требует HappyWhale + GBIF + iNaturalist connector
- **Claimed:** Параметр 6 выполнен, 4 sinks, ≥ 2 БД + ≥ 2 платформы.
- **Actual:**
  1. 2 БД — ок: `postgres_sink.py` + `sqlite_sink.py`.
  2. 2 «платформы мониторинга» — спорно. `/metrics` (Prometheus) — мониторинг, `otel_sink.py` — мониторинг. **Но это мониторинг ВАШЕГО сервиса**, а не внешняя платформа мониторинга морских млекопитающих. ТЗ §Параметр 6 упоминает «платформ мониторинга уровня HappyWhale», то есть **биологические** платформы (HappyWhale, GBIF, OBIS, iNaturalist), а не DevOps-телеметрия.
  3. PLAN 3.4 явно требует `integrations/happywhale_sink/connector.py`, `integrations/gbif_sink.py`, `integrations/inat_sink.py` — **ни одного нет**. INTEGRATION_GUIDE.md §5 признаёт это: «darwin_core_sink.py is on the roadmap for Q3 2026».
  4. `GRANT_DELIVERABLES.md:18` list включает «HF Hub» — это model registry, не платформа мониторинга.
- **Source:** TЗ §Параметр 6 («совместимость и обмен данными не менее чем с 2 базами данных и 2 ведущими платформами мониторинга уровня HappyWhale»), PLAN_STAGE3.md §3.4
- **Fix:**
  1. Реализовать **как минимум** один коннектор к HappyWhale (через https://happywhale.com/api если есть, или хотя бы через CSV-экспорт в их формат) + один к GBIF Darwin Core.
  2. Или формально согласовать с ФСИ другую интерпретацию «платформы» (Prometheus + OpenTelemetry) — письменно.
  3. Обновить INTEGRATION_GUIDE.md, убрать «Q3 2026» roadmap заметку, показать рабочий код.

---

#### [F5] Залежалые notebook пути — Экспертиза 2.0 §2.1 требует возможности запуска всех ipynb
- **Where:**
  - `research/notebooks/07_onnx_inference_compare.ipynb:207,237` — `torch.load('./models/model-e15.pt', …)` (см. `grep`)
  - `research/notebooks/12_test_detection_id.ipynb:16,65,62336,62406,62516,62526,62660,62683,62739,62908` — абсолютные пути `/Users/savandanov/Documents/Github/whales-identification/...` (≥ 10 вхождений)
  - PLAN_STAGE3.md §1.3 (помечено 🟠 high) — не исправлено
- **Claimed:** Все ipynb запускаемы, пути относительные, ссылки обновлены.
- **Actual:**
  1. Notebook 07 в cell 0 теперь имеет пометку «Данный ноутбук работает с legacy ViT-моделью `model-e15.pt`. Модель **не скачивается автоматически** через download_models.sh — нужно взять с Yandex Disk.» — **но** cell 207 и 237 всё равно пытаются `torch.load('./models/model-e15.pt')`. Без этой модели ячейки упадут. Экспертиза 2.0 §4.2 уже указывала на эту проблему.
  2. Notebook 12 содержит минимум 10 абсолютных путей к `/Users/savandanov/…`. Запуск невозможен на любой другой машине. Это ≥ 3-й раз, что указывается в экспертизе.
- **Source:** Экспертиза 2.0 §2.1, §4.2, PLAN_STAGE3.md §1.3
- **Fix:**
  1. 07_onnx_inference_compare.ipynb: либо скачать модель через entrypoint (добавить запись в `scripts/download_models.sh` для Yandex Disk — сейчас нет), либо переписать cells на `efficientnet_b4_512_fold0.ckpt`.
  2. 12_test_detection_id.ipynb: заменить все `/Users/savandanov/…` на `..` относительные пути, через `pathlib.Path(__file__).parent` или `os.environ['REPO_ROOT']`.

---

### 🟠 High

#### [F6] Жирный блок фабрикованных цифр в README.md:383-398 — та же проблема, за которую уже получили замечание round 3
- **Where:** `README.md:387-398` (таблица «Результаты сравнения»)
- **Claimed:** Precision 82–93 %, TPR 76–91 %, TNR 88–94 %, F1 0.79–0.92, Availability 90–95 % для 7 моделей, «~60 000 train / ~20 000 test» для каждой.
- **Actual:** Ни одна цифра не воспроизводится никаким скриптом в репозитории. `compute_metrics.py` даёт только одну модель (effb4). В контексте round 3 КП 1 «random.uniform(0.85, 0.95)» эта таблица выглядит точно так же — красивые рандомные числа без подтверждения.
- **Fix:** Заменить таблицу на реальные результаты `compute_metrics.py --model ...` для каждой модели (если `models_config.yaml` поддерживает переключение), или удалить 6 строк и оставить только effb4 с подтверждёнными цифрами, или честно пометить «референсные данные из литературы».

---

#### [F7] wiki_content/Model-Cards.md содержит фабрикованные per-species метрики и `model-e15.pt`
- **Where:** `wiki_content/Model-Cards.md:78,88-96,100-111,154`
- **Claimed:** «Humpback Whale: Precision 95.3 %, Recall 93.8 %, Sample Count 12543» и т. д. для 10 видов.
- **Actual:** Эти цифры нигде не вычисляются. Sample Count 12543 для Humpback Whale — взято из внешних happy-whale-and-dolphin competition EDA, не из реального test split проекта. Также 6 раз упоминается устаревший `model-e15.pt`.
- **Fix:** Полностью переписать Model-Cards.md на фактический effb4-arcface-v1 с цифрами из `reports/metrics_latest.json`. Удалить per-species таблицу или заменить на species_map.csv (30 видов без precision).

---

#### [F8] Противоречивые числа в `DOCS/GRANT_DELIVERABLES.md` vs `reports/metrics_latest.json`
- **Where:**
  - `DOCS/GRANT_DELIVERABLES.md:13` — «**93.55 %**, n=60»
  - `DOCS/GRANT_DELIVERABLES.md:14` — «p95 = **540 ms**»
  - `DOCS/GRANT_DELIVERABLES.md:15` — «R² = **0.9982** на точках [5, 10, 20, 30]»
  - `DOCS/GRANT_DELIVERABLES.md:16` — «≤ **6.9 %**»
  - `DOCS/GRANT_DELIVERABLES.md:21` — «TNR **93.33 %**»
  - `reports/metrics_latest.json` — Precision 0.9048 (n=202), p95 519.42 ms, R²=1.0 на [10, 25, 50, 100], TNR 0.902, noise drop 0.0 %
- **Claimed:** Все параметры выполнены.
- **Actual:** Числа в GRANT_DELIVERABLES взяты из **старого запуска** (видимо, до пересчёта в текущей сессии), числа в reports/ — актуальные. Разница 0.9048 vs 0.9355 — несколько процентных пунктов, эксперт это заметит.
- **Source:** round 3 замечание о консистентности цифр между документами
- **Fix:** Перегенерировать GRANT_DELIVERABLES.md из `reports/metrics_latest.json` одним прогоном (сделать скрипт `scripts/update_grant_deliverables.py`).

---

#### [F9] Противоречивые числа в `DOCS/ML_ARCHITECTURE.md:127` — третий вариант
- **Where:** `DOCS/ML_ARCHITECTURE.md:127,136`
- **Claimed:** Precision (PPV) 0.9355; Top-1 accuracy 0.1667 (5/30).
- **Actual:** Ни 5/30 (16.67 %), ни 22/100 (22 %) не совпадают с `reports/metrics_latest.json` → 22/100. И precision 0.9355 ≠ 0.9048.
- **Fix:** Обновить ML_ARCHITECTURE.md на актуальные цифры; привязать к `<!-- metrics:start --><!-- metrics:end -->` маркерам, которые `compute_metrics.py --update-model-card` уже пишет в MODEL_CARD.md.

---

#### [F10] `models_config.yaml::vit_l32.checkpoint = "models/model-e15.pt"` — файл не скачивается
- **Where:** `models_config.yaml:26`
- **Claimed:** Конфигурация pluggable, ViT-L/32 — «legacy, deprecated» но запись оставлена для Stage 1.
- **Actual:** PLAN §1.4 помечен 🟡 medium. Помечен как deprecated, но: (a) `models/registry.json::active == vit_l32` (см. F11); (b) download_models.sh не скачивает model-e15.pt; (c) при переключении `active_model: vit_l32` пайплайн упадёт на `torch.load`.
- **Fix:** Либо убрать запись из `models_config.yaml`, либо добавить опциональный скачиватель (Yandex Disk как в 07 notebook), либо явно `if deprecated: raise`.

---

#### [F11] `models/registry.json::active = vit_l32` vs `models_config.yaml::active_model = effb4_arcface`
- **Where:**
  - `models/registry.json:3` — `"active": "vit_l32"`
  - `models/registry.json:11` — `"weights_path": "models/model-e15.pt"`
  - `models_config.yaml:6` — `active_model: "effb4_arcface"`
- **Claimed:** Model Registry единый источник правды для production модели.
- **Actual:** Два источника, разные значения. Непонятно, какой источник эксперт должен смотреть. Код пайплайна вообще использует `_load()` с hard-coded приоритетом effb4 > vit > resnet, **игнорируя оба файла**. Это то самое «непонятно как выбирается модель», что round 4 КП 3.1/4.1 уже указывало.
- **Fix:** Обновить `registry.json::active = "effb4_arcface"`, добавить `effb4_arcface` как элемент `models[]` с `sha256` и `metrics_snapshot = reports/metrics_latest.json`. Либо удалить `models_config.yaml` и оставить registry.json как единственный источник.

---

#### [F12] `model_version: "vit_l32-v1"` hardcoded в 4 файлах — реально грузится `effb4-arcface-v1`
- **Where:**
  - `whales_be_service/src/whales_be_service/main.py:219,233` — `DETECTION_EXAMPLE`, `REJECTION_EXAMPLE`
  - `whales_be_service/src/whales_be_service/inference/identification.py:53` — default parameter
  - `whales_be_service/src/whales_be_service/inference/pipeline.py:65` — docstring
  - `whales_be_service/src/whales_be_service/response_models.py:34` — `Detection.model_version: str = "vit_l32-v1"`
  - `MODEL_CARD.md:123`
- **Claimed:** model_version reflects actual loaded backend (docstring in pipeline.py:64).
- **Actual:** Фактически у запущенного пайплайна `model_version == "effb4-arcface-v1"` (см. `identification.py:91`). Но в openapi examples, в Detection default, в MODEL_CARD — везде "vit_l32-v1". Эксперт увидит "vit_l32-v1" в Swagger examples и сделает вывод, что production использует ViT.
- **Fix:** Заменить все 4 места на `effb4-arcface-v1`, либо сделать default полем, которое заполняется на старте из `get_pipeline().model_version`.

---

#### [F13] `wiki_content/Contributing.md:748` — «Models: Apache 2.0 (with restrictions)» — прямо запрещено Экспертизой 2.0 §1.1
- **Where:** `wiki_content/Contributing.md:748`
- **Claimed:** Трёхуровневая модель CC-BY-NC-4.0 для моделей.
- **Actual:** Одно упоминание Apache 2.0 для моделей в Contributing.md остаётся. Экспертиза 2.0 §1.1 явно: «Недопустимо использовать формулировку Apache License 2.0 применительно к … LICENSE_MODELS.md».
- **Fix:** Заменить на `CC-BY-NC-4.0`.

---

#### [F14] `huggingface/README.md:84` — «Apache License 2.0 with Usage Restrictions» — та же проблема
- **Where:** `huggingface/README.md:84`
- **Actual:** `huggingface/README.md` — шаблон для Hugging Face card, используется как источник для обновления HF (см. `scripts/update_huggingface.sh`). Если скрипт запустят, неправильная лицензия попадёт на HF. Там уже сейчас «MIT» (Экспертиза 2.0 §1.1.1). Нужна согласованность.
- **Fix:** Заменить на `cc-by-nc-4.0` по всему `huggingface/`. Проверить актуальное состояние на HuggingFace (https://huggingface.co/0x0000dead/ecomarineai-cetacean-effb4) — там должно быть `cc-by-nc-4.0`.

---

#### [F15] `frontend/src/api.ts:2` — `VITE_BACKEND ?? 'http://localhost:8000'` — Экспертиза 2.0 §1.2.4 повторное замечание
- **Where:** `frontend/src/api.ts:2`
- **Claimed:** Исправлено через `ALLOWED_ORIGINS` + runtime warning + документация `VITE_BACKEND + --no-cache` (Wiki/Installation.md:128-140).
- **Actual:**
  1. Дефолт `http://localhost:8000` остаётся. При запуске на другом хосте без явного `VITE_BACKEND` фронт будет ломаться ровно как описано экспертом.
  2. Warning в консоли — не решение. ФСИ-эксперт использует Docker compose на одной машине и открывает UI с другой — **та самая проблема**.
  3. Правильный подход: фронт должен использовать `window.location.origin:8000` или relative `/api`, либо Nginx прокси.
- **Source:** Экспертиза 2.0 §1.2.4, повторное (round 4)
- **Fix:** Заменить дефолт на `\`http://${window.location.hostname}:8000\`` или на `/api` с nginx upstream. Или задокументировать как единственный метод `docker compose build --build-arg VITE_BACKEND=http://<host>:8000` и проверить, что этот путь работает end-to-end.

---

#### [F16] PLAN §3.1 Заключительный НТО и 4 руководства **не созданы**
- **Where:** `docs_fsie/` — не существует
- **Claimed:** PLAN 3.1 deliverables: заключительный НТО (≥50 стр., ГОСТ 7.32-2017), 4 руководства (пользователя, админа, разработчика, контрибьютора) как docx.
- **Actual:** Ничего нет. `DOCS/` содержит `USER_GUIDE_BIOLOGIST.md` (похоже на руководство пользователя), `DEPLOYMENT.md` (админа), `ML_ARCHITECTURE.md` + `API_REFERENCE.md` (разработчика), `wiki_content/Contributing.md` (контрибьютора). Но это: (a) не docx, (b) не по ГОСТ 7.32-2017, (c) не единый документ, (d) не подписаны, (e) нет списка исполнителей с ролями.
- **Source:** PLAN_STAGE3.md §3.1, `work/stage3/extracted_docs/инструкция_заключительная.md`
- **Fix:** Сгенерировать `docs_fsie/` с 5 документами: Заключительный_НТО.docx, Руководство_пользователя.docx, Руководство_администратора.docx, Руководство_разработчика.docx, Руководство_контрибьютора.docx. Использовать ГОСТ 7.32-2017 шаблон (структурные элементы, нумерация, шрифт Times New Roman 14, 1.5 интервал). Минимум 50 стр. основной части в НТО.

---

### 🟡 Medium

#### [F17] PLAN §3.2 k8s — отсутствует целиком
- **Where:** `k8s/` — не существует
- **Claimed:** `k8s/deployment.yaml` (replicas: 3), `k8s/hpa.yaml` (CPU > 70% → scale до 10), `k8s/ingress.yaml` (60 req/min rate limiting), `reports/LOAD_TEST.md` с 50 RPS locust.
- **Actual:** Ничего. `tests/performance/locustfile.py` есть, но LOAD_TEST.md с результатами 50 RPS — нет.
- **Fix:** Создать `k8s/` файлы + запустить locust с `users=50, spawn-rate=5, host=http://localhost:8000`, зафиксировать p95 в `reports/LOAD_TEST.md`.

---

#### [F18] PLAN §3.3 Colab notebook + скринкаст — отсутствуют
- **Where:** `docs/QUICKSTART_COLAB.ipynb` — не существует; скринкаст mp4 — не существует
- **Claimed:** Google Colab ноутбук «Быстрый старт» + MP4 3–5 мин.
- **Actual:** Ничего.
- **Fix:** Создать Colab ipynb с `!pip install`, `!curl -X POST` к публичному эндпоинту. Скринкаст — опционально, но PLAN явно требует.

---

#### [F19] PLAN §3.5 — `10_hyperparameter_search.ipynb`, INT8 quantization, `reports/OPTIMIZATION.md` — отсутствует
- **Where:** `research/notebooks/10_*` — есть `10_remove_bg_example.ipynb`, нет `10_hyperparameter_search.ipynb`; `reports/OPTIMIZATION.md` — нет.
- **Claimed:** Grid search по min_confidence (0.05/0.10/0.15) и CLIP threshold (0.45/0.55/0.65) + INT8 quantization EffB4.
- **Actual:** Ничего.
- **Fix:** Добавить `13_hyperparameter_search.ipynb` (или переименовать 10_remove_bg чтобы освободить номер) + INT8 quantization prototype + `reports/OPTIMIZATION.md`.

---

#### [F20] PLAN §3.6 — `11_ensemble_architecture.ipynb` + ensemble mode в pipeline + `ENSEMBLE.md` — отсутствует
- **Where:** `research/notebooks/11_*` — есть `11_data_stram_cv_video.ipynb`, нет `11_ensemble_architecture.ipynb`; pipeline в `inference/identification.py` не поддерживает `mode: ensemble`; `reports/ENSEMBLE.md` — нет.
- **Claimed:** Ensemble EffB4 + CLIP gate + YOLOv8 bbox с замером точности/latency.
- **Actual:** Ничего. В `models_config.yaml` нет секции ensemble.
- **Fix:** Реализовать или явно отметить как «не реализовано, потому что текущий single model уже покрывает ТЗ».

---

#### [F21] PLAN §3.7 — `docker-compose.prod.yml` + `scripts/start.sh` + `start.bat` — отсутствуют
- **Where:** Все три файла — не существуют.
- **Claimed:** Production compose (resource limits, restart: always, логи volume), скрипт start.sh/start.bat для Linux/Mac/Windows.
- **Actual:** Ничего.
- **Fix:** Написать 3 файла.

---

#### [F22] PLAN §3.8 — Webhook `POST /v1/webhook/register` + `GET /v1/export?format=csv|json` — **не реализованы**
- **Where:** `whales_be_service/src/whales_be_service/main.py` — нет endpoint'ов webhook, export
- **Claimed:** Endpoint регистрации callback + экспорт истории.
- **Actual:** `grep webhook` — 0 совпадений в исходниках.
- **Fix:** Добавить два endpoint в `main.py` + хранилище истории (SQLite persistent volume).

---

#### [F23] `wiki_content/` — устаревший `model-e15.pt` в 6 файлах
- **Where:** `wiki_content/Model-Cards.md:78,154`; `wiki_content/Usage.md:327,350`; `wiki_content/FAQ.md:116,125,145,148`; `wiki_content/Architecture.md:60,455,549`; `wiki_content/Testing.md:241,693`; `wiki_content/Installation.md:275,279,283`
- **Claimed:** Wiki актуализирован, пайплайн = EfficientNet-B4.
- **Actual:** Пятна устаревших путей остаются. Экспертиза 2.0 §1.2.2.2 и §4.2 прямо на это указывали.
- **Fix:** Автозамена `model-e15.pt` → `efficientnet_b4_512_fold0.ckpt` везде кроме wiki_content/Installation.md §275-283 (там явная документация legacy demo-ui) — и пометить её как «для legacy Streamlit демо; production использует effb4».

---

#### [F24] `MODEL_CARD.md` header блок ещё про ViT-L/32 — не обновлён под EffB4
- **Where:** `MODEL_CARD.md:6-8`
- **Claimed:** CLIP + ViT-L/32 pipeline.
- **Actual:** Production pipeline = CLIP + EfficientNet-B4 ArcFace (см. `identification.py:83-92`). Заголовок MODEL_CARD — устаревший.
- **Fix:** Обновить заголовок: «EcoMarineAI EfficientNet-B4 ArcFace (identification) + CLIP ViT-B/32 (anti-fraud)».

---

#### [F25] `config.yaml::15587 классов` vs реально `13837` — несоответствие
- **Where:**
  - `whales_be_service/src/whales_be_service/config.yaml`, `config_full.yaml`, `config_full_renamed.yaml`
  - `models_config.yaml::vit_l32.num_classes: 15587` vs `effb4_arcface.num_classes: 13837`
  - CLAUDE.md — «15 587 individual whale IDs»
  - `MODEL_CARD.md:29` — «13 837 active»
- **Claimed:** 15 587 классов.
- **Actual:** В активном чекпоинте effb4 обучены только 13 837 индивидов (MODEL_CARD:29 явно пишет «1 750 unused slots»). Но main.py:215 и Detection response schema всё ещё говорят про 15 587.
- **Fix:** Унифицировать: либо везде 13 837, либо явно «15 587 slots, 13 837 used». CLAUDE.md нужно обновить.

---

### ⚪ Low / Nits

#### [F26] README.md:19 — «Precision 0.905» — должна быть явная пометка «anti-fraud gate»
- **Where:** `README.md:19`
- **Fix:** Таблица должна иметь заголовок «CLIP anti-fraud gate metrics» с отдельной пометкой, что Individual ID precision отличается.

#### [F27] `MODEL_CARD.md` bottom block ещё показывает `model-e15.pt` **Note** block
- **Where:** `MODEL_CARD.md` — искать внизу legacy notes
- **Fix:** Убрать или явно пометить «legacy v1.0.0».

#### [F28] `wiki_content/Home.md:14` — «1000 индивидуальных особей» — несоответствие 13 837
- **Where:** `wiki_content/Home.md:14`
- **Fix:** Заменить на «13 837 индивидуальных особей (≥ 1000 по ТЗ)».

#### [F29] `wiki_content/Home.md:73-79` — таблица 5 моделей с фабрикованными цифрами (Precision 85–93 %)
- **Where:** `wiki_content/Home.md:73-79`
- **Fix:** То же что F6/F7 — удалить/обновить.

#### [F30] `LICENSE_MODELS.md:132-136` — таблица моделей ещё показывает `model-e15.pt`, `resnet101.pth`, `efficientnet-b5.pth` без упоминания актуальной effb4_arcface
- **Where:** `LICENSE_MODELS.md:131-137`
- **Fix:** Добавить строку для `efficientnet_b4_512_fold0.ckpt`.

#### [F31] `ci.yml` — job `test` имеет `needs: [lint]` но нет `needs: [lint, security]`
- **Where:** `.github/workflows/ci.yml:96`
- **Actual:** Security scan может не запуститься, если lint упал. Но `security` сам не зависит от lint, это ок. Проблема косметическая — последовательность стадий в документации говорит `lint → test → security → build`, фактически `lint → (test, security) → docker`. Нужно обновить комментарий «Stage N:» в workflow.
- **Fix:** Обновить комментарии чтобы отражали реальный DAG.

---

## Coverage check: 13 параметров ТЗ

| # | Параметр | Целевое | Статус | Где доказательство | Replay cmd |
|---|----------|---------|--------|-------------------|------------|
| 1 | Precision идентификации | ≥ 80 % | **❌** | `reports/metrics_latest.json::identification.top1_accuracy = 0.22` (22 %) — на 3.6× ниже порога. Все 88/90.48/93.55 % — precision **бинарного анти-фрод гейта**, не §Параметр 1. | `python scripts/compute_metrics.py` |
| 2 | Скорость обработки | ≤ 8 с | ✅ | `reports/metrics_latest.json::performance.latency_p95_ms = 519.42` | тот же скрипт |
| 3 | Масштабируемость (линейная) | линейная | ✅ | `reports/scalability_latest.json::regression.r_squared = 1.0` (slope 0.482 s/image) | `python scripts/benchmark_scalability.py` |
| 4 | Универсальность / шум | drop ≤ 20 % | ✅ | `reports/NOISE_ROBUSTNESS.md` — max drop −1.1 % | `python scripts/benchmark_noise.py` |
| 5 | Интерфейс | мин. обучение | ⚠️ | React UI + Streamlit + CLI + Swagger — есть. Но фронт ломается при non-localhost из-за F15. | `docker compose up` |
| 6 | Интеграция | ≥ 2 БД + ≥ 2 платформы | **⚠️ → ❌** | SQLite + Postgres ✓ (2 БД). OTel + Prometheus — это мониторинг СВОЕГО сервиса, а не платформы уровня HappyWhale. Биологических коннекторов (HappyWhale/GBIF/iNat) нет. См. F4. | `python integrations/sqlite_sink.py …` |
| 7 | Availability ≥ 95 % / 7 дней | 95 % / 7 д | **❌** | Только `availability_percent` gauge в памяти процесса. Нет production деплоя, нет 7-дневного замера. См. F3. | `curl /metrics` |
| 8 | Sensitivity (TPR) | > 85 % | ✅ (только анти-фрод) | `reports/metrics_latest.json::anti_fraud.tpr = 0.95`. Но это TPR бинарного гейта, не идентификации. | `compute_metrics.py` |
| 9 | Specificity (TNR) | > 90 % | ✅ (только анти-фрод) | `anti_fraud.tnr = 0.902` | тот же |
| 10 | Recall | > 85 % | ✅ (= TPR) | 0.95 | тот же |
| 11 | F1 | > 0.6 | ✅ (только анти-фрод) | `anti_fraud.f1 = 0.9268` | тот же |
| 12 | Датасет 80 k / 1 k особей | 80 k / 1 k | **⚠️** | Public train 51 k image. Ministry RF «private, not redistributable». Evaluation split — только 202 image. 13 837 индивидов ≥ 1 000 ✓, но 51 k < 80 k. См. F2. | `cat MODEL_CARD.md` |
| 13 | Объекты: киты + дельфины | киты + дельфины | ✅ | `species_map.csv` → 30 видов (humpback, blue, fin, beluga, killer, bottlenose dolphin, etc.). | `wc -l whales_be_service/src/whales_be_service/resources/species_map.csv` |

**Summary: 6 ✅ явно выполнено (2,3,4,5,10,13), 3 ✅ только для анти-фрод гейта (8,9,11), 1 ⚠️ сомнительно (12), 3 ❌ не выполнено (1,6,7).** Параметры 1 и 7 — блокеры для сдачи; Параметр 6 — под вопросом интерпретации «платформа мониторинга».

---

## Coverage check: 8 работ Stage 3 (3.1–3.8 по PLAN)

| # | Работа | Статус | Артефакт | Gap |
|---|--------|--------|----------|-----|
| 3.1 | Итоговая тех. документация (Балцат) | ❌ | Нет `docs_fsie/` с Заключительным НТО + 4 руководствами | Всё предстоит сделать. DOCS/ содержит разрозненные .md, не соответствующие ГОСТ 7.32-2017 |
| 3.2 | MLOps: масштабирование (Балцат) | ❌ | `k8s/` нет, `reports/LOAD_TEST.md` нет | Реализовать k8s deploy + hpa + ingress + запустить locust |
| 3.3 | Учебные/демо материалы (Балцат) | ❌ | `docs/QUICKSTART_COLAB.ipynb` нет, скринкаст нет | Создать ipynb + записать видео |
| 3.4 | API для интеграций (Ванданов) | ⚠️ | `integrations/postgres_sink.py` + `sqlite_sink.py` + `otel_sink.py` есть | **Нет HappyWhale/GBIF/iNat коннекторов** — PLAN явно требует, INTEGRATION_GUIDE признаёт как roadmap |
| 3.5 | Оптимизация параметров (Ванданов) | ❌ | Notebook отсутствует, `reports/OPTIMIZATION.md` нет | Grid search по min_confidence/CLIP threshold + INT8 quantization |
| 3.6 | Комплексная CV архитектура (Серов) | ❌ | `11_ensemble_architecture.ipynb` нет, `ensemble` mode в identification.py нет, `reports/ENSEMBLE.md` нет | Реализовать или обосновать отказ |
| 3.7 | Контейнеризация (Тарасов) | ❌ | `docker-compose.prod.yml` нет, `start.sh`/`start.bat` нет | Написать 3 файла + smoke через docker profile |
| 3.8 | Интеграция с внешними сервисами (Тарасов) | ❌ | Нет webhook endpoint, нет export endpoint, `INTEGRATION_GUIDE.md` частично обновлён | Добавить 2 endpoint + persistent storage истории |

**Stage 3 Score: 0/8 полностью выполнено.** Stage 3 работ ещё не начинался — только Stage 2 исправления довели до текущего состояния.

---

## Next actions (для fixer-агента)

Порядок по приоритету (сначала блокёры, потом высокие, потом среднее):

1. **[F1 критический] Пересчитать Параметр 1.**
   - Добавить в `scripts/compute_metrics.py` метрику `identification.precision_clear = правильных top-1 / все accepted cetacean изображения ≥ laplacian_threshold`.
   - Применить fold 0 фильтр если чекпоинт effb4 fold 0 only — записать в `evaluation_caveats` секцию `metrics_latest.json`.
   - Либо пересчитать на species-level (по species_map.csv) — это вероятнее даст ≥ 80 %.
   - Обновить MODEL_CARD, README, GRANT_DELIVERABLES, PERFORMANCE_REPORT, SOLUTION_OVERVIEW, ML_ARCHITECTURE с **разделением** `anti_fraud_precision` vs `identification_precision`.
   - Если обе метрики не дотягивают, явно признать в справке об устранении, что §Параметр 1 требует обсуждения интерпретации с ФСИ.

2. **[F3 критический] Параметр 7 Availability.**
   - Поднять бэкенд на публичный URL через Render.com/Fly.io (free tier) минимум 7 дней.
   - Подключить Better Stack / UptimeRobot на /health endpoint, интервал 1 мин.
   - Приложить CSV с uptime за неделю в Заключительный НТО §7.

3. **[F4 критический] Параметр 6 платформы.**
   - Реализовать `integrations/gbif_sink.py` — экспорт в Darwin Core CSV формат.
   - Реализовать `integrations/happywhale_submit.py` — хотя бы JSON POST-заглушка с документацией (если API недоступен без ключа).
   - Обновить INTEGRATION_GUIDE.md, удалить «roadmap Q3 2026» note.

4. **[F5 критический] Notebook пути.**
   - `research/notebooks/07_onnx_inference_compare.ipynb`: заменить все `torch.load('./models/model-e15.pt')` на effb4 либо добавить скачиватель Yandex Disk.
   - `research/notebooks/12_test_detection_id.ipynb`: автозамена `/Users/savandanov/Documents/Github/whales-identification/` → `..` (репо-относительный путь).

5. **[F2 критический] Параметр 12 датасет.**
   - В Заключительный НТО: честное описание public 51 k + Ministry RF «не использовалась в чекпоинте / использовалась с audit trail».
   - Расширить test split до ≥ 1000 image (добавить больше positives из HappyWhale Kaggle, negatives из Intel Images или COCO).

6. **[F16 critical] Создать `docs_fsie/` по ГОСТ 7.32-2017.**
   - Template Заключительного НТО с: списком исполнителей, рефератом, содержанием, нормативными ссылками, определениями, обозначениями, 7 разделами по задачам 3.1–3.8, заключением, списком источников, приложениями.
   - 4 руководства как отдельные .docx.

7. **[F11] `models/registry.json`.** Заменить `active = "effb4_arcface"` и добавить запись с sha256.

8. **[F12] `model_version` string.** Заменить 4 вхождения `vit_l32-v1` на `effb4-arcface-v1`.

9. **[F13, F14] Apache 2.0 residuals.** Заменить 2 упоминания в `wiki_content/Contributing.md` и `huggingface/README.md` на CC-BY-NC-4.0.

10. **[F15] frontend/src/api.ts default.** Заменить `'http://localhost:8000'` на динамический `\`http://${window.location.hostname}:8000\`` с warning только при reset-to-default.

11. **[F6, F7, F23, F29] Фабрикованные числа и model-e15.pt в wiki.**
    - `README.md:383-398` — таблица 7 моделей: заменить на реальные воспроизводимые числа либо пометить как литературные бенчмарки.
    - `wiki_content/Model-Cards.md` — переписать под effb4 только, удалить per-species фабрикованные precision.
    - `wiki_content/Home.md:73-79` — удалить/обновить.
    - Автозамена `model-e15.pt` в Usage, FAQ, Architecture, Testing, Model-Cards wiki.

12. **[F8, F9] Противоречивые числа в GRANT_DELIVERABLES и ML_ARCHITECTURE.**
    - Сделать `scripts/update_grant_deliverables.py`, который регенерирует оба файла из `reports/metrics_latest.json`.
    - Перегенерировать после F1 пересчёта.

13. **[F10] `models_config.yaml::vit_l32.checkpoint`.** Либо удалить запись vit_l32 полностью, либо заменить checkpoint на Yandex Disk URL с комментарием «optional, download manually».

14. **[F24] MODEL_CARD.md заголовок.** Обновить Architecture поле: «EfficientNet-B4 (identification) + CLIP ViT-B/32 (anti-fraud gate)».

15. **[F25] 15587 vs 13837.** Унифицировать по всем .md + CLAUDE.md → `13 837 active / 15 587 head slots`.

16. **[F17] k8s.** Написать `k8s/deployment.yaml`, `k8s/hpa.yaml`, `k8s/ingress.yaml`, запустить locust, сгенерировать `reports/LOAD_TEST.md`.

17. **[F18] Colab notebook.** Создать `docs/QUICKSTART_COLAB.ipynb` с ячейками `!pip install requests`, `!curl -X POST …`.

18. **[F19] Оптимизация.** Создать `research/notebooks/13_hyperparameter_search.ipynb` + `reports/OPTIMIZATION.md`.

19. **[F20] Ensemble.** Создать `research/notebooks/14_ensemble_architecture.ipynb` + `reports/ENSEMBLE.md` либо явно обосновать отказ.

20. **[F21] Prod Docker.** Написать `docker-compose.prod.yml` (resource limits, restart: always, volumes:logs), `scripts/start.sh`, `scripts/start.bat`.

21. **[F22] Webhook/Export endpoints.** Добавить `POST /v1/webhook/register` и `GET /v1/export?format=csv|json` в `main.py` + persistent storage истории через sqlite в app.state.

22. **[F30] LICENSE_MODELS.md таблица моделей.** Добавить строку `efficientnet_b4_512_fold0.ckpt`.

23. **[F27, F28] Wiki nit'ы.** Убрать legacy model-e15.pt внизу MODEL_CARD, обновить Home.md:14 «1000 индивидуальных особей → 13837».

24. **GitHub Discussions + Pages redirect.** Требуют access к GitHub Settings репозитория `0x0000dead/whales-identification` — это ручное действие владельца, не код. Упомянуть в справке как «выполнено на стороне GitHub Settings» и приложить скриншоты Settings.

25. **Справка об устранении замечаний round 5.** После всех выше — сформировать новый документ `work/stage3/fixes/Справка_round5.md` со всеми 11 пунктами Экспертизы 2.0, привязанными к конкретным файлам/commit'ам, которые закрыли каждое замечание.

26. **Перезапустить `scripts/compute_metrics.py --update-model-card`** после F1 fix — чтобы цифры в MODEL_CARD между `<!-- metrics:start -->` и `<!-- metrics:end -->` отражали пересчитанную identification_precision.

27. **Добавить pytest-coverage report в CI артефакты.** Сейчас `continue-on-error: true` на codecov upload в ci.yml — уберёт любую информацию о coverage. Для ФСИ требуется покрытие ≥ 80 % (CLAUDE.md cat 5.1).

28. **Проверить `models/registry.json` sha256: null → заполнить.** Требование СП 2.1 «хеширование» — сейчас нулевые значения.

29. **Sanity-check: запустить смоуку end-to-end.** `docker compose up → POST /v1/predict-single (positive image) → POST /v1/predict-single (negative image)` → проверить `model_version` в ответе, `cetacean_score`, `probability`. Убедиться, что full pipeline работает без каких-либо внешних ресурсов.

30. **Code freeze → tag v2.0.0 → docker push → обновить HF model card → сгенерировать Заключительный НТО.**

---

## Notes for reviewer of this audit

- Я не проверял полноту GitHub Actions workflow runs (round 4 КП 2.25.2 требовал ссылки на конкретные run’ы). Это внешний артефакт, невозможно проверить локально.
- Я не проверял GitHub Pages / Wiki live state — `vandanov.company` redirect и Discussions включение требуют access к репо Settings.
- Я не проверял `huggingface.co/0x0000dead/ecomarineai-cetacean-effb4` live license — требует WebFetch.
- Метрики в `reports/metrics_latest.json` я принял на веру, не запускал `compute_metrics.py` на реальных данных (audit это чтение, не execution).
- CLAUDE.md утверждает Общая оценка зрелости 90/100 — по результату аудита реальная готовность к сдаче Stage 3 оценивается как **60/100** (Stage 2 исправления частично есть, Stage 3 не начат).
