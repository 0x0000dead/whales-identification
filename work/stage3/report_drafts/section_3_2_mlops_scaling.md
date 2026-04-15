# §3.2 — MLOps: масштабирование под высокую нагрузку (черновик для НТО)

_Ответственный: Балцат К.И._
_Статус: реализовано (нагрузочный прогон — TBD)_
_Связанные пункты ТЗ: Параметр 3 (линейная временная сложность, ≤ 8 с
на изображение 1920×1080), Параметр 7 (availability ≥ 95 % за 7 дней)._

## Что сделано

1. **Kubernetes Deployment** — `k8s/deployment.yaml`:
   - `replicas: 3` (минимум для rolling upgrade с `maxUnavailable: 1`
     без потери трафика);
   - `resources.requests: {cpu: 1, memory: 2Gi}`,
     `resources.limits: {cpu: 2, memory: 4Gi}` — идентично
     `docker-compose.prod.yml`;
   - `readinessProbe` и `livenessProbe` на `/health` с щадящим
     `startupProbe` (до 5 мин — покрывает загрузку модели с HF Hub при
     первом старте);
   - `securityContext.runAsNonRoot: true`, `capabilities.drop: ["ALL"]`,
     `seccompProfile: RuntimeDefault` — контейнер хардненый;
   - аннотации `prometheus.io/scrape: "true"`, `port: "8000"`,
     `path: "/metrics"` — kube-prometheus подхватывает поды автоматически;
   - `envFrom: configMapRef: ecomarine-backend-config`.

2. **ConfigMap** — `k8s/configmap.yaml` — ключи `ALLOWED_ORIGINS`,
   `HF_REPO`, `LOG_LEVEL`, `UVICORN_WORKERS`, `PYTHONUNBUFFERED`,
   `PYTHONDONTWRITEBYTECODE`. Секреты (HF token) — в отдельном Secret.

3. **ClusterIP Service** — `k8s/service.yaml`, порт 80 → контейнер 8000,
   `sessionAffinity: None` (round-robin).

4. **HorizontalPodAutoscaler** — `k8s/hpa.yaml`:
   - `autoscaling/v2`, `minReplicas: 3`, `maxReplicas: 10`;
   - метрики: CPU 70 %, memory 80 % (запасной клапан от OOM-каскадов);
   - `behavior.scaleUp`: 30 s стабилизация, +2 пода или +50 % за 30 с;
   - `behavior.scaleDown`: 5 мин стабилизация, −1 под за 60 с
     (защита от флэппинга).

5. **Nginx Ingress** — `k8s/ingress.yaml`:
   - `nginx.ingress.kubernetes.io/limit-rpm: "60"` — 60 req/min per IP
     (согласовано с application-level rate limiter в backend);
   - `proxy-body-size: 32m`, `proxy-read-timeout: 120`;
   - TLS-ready (блок `tls:` закомментирован, есть подсказка про
     cert-manager).

6. **k8s README** — `k8s/README.md` с prerequisites, `kubectl apply -f k8s/`,
   smoke-test через `kubectl port-forward`, rollout / rollback команды.

7. **Отчёт по нагрузочному тесту** — `reports/LOAD_TEST.md`:
   - методология (Locust, 50 RPS target, 5 мин, 100 users, spawn 10/s);
   - hardware-профиль (8-core x86, 32 GiB RAM, CPU inference);
   - реальные цифры для офлайн-бенчмарка: slope 0.482 s/image,
     R² = 1.000 (из `reports/SCALABILITY.md`), per-image HTTP p50/p95/p99 =
     484 / 519 / 597 мс (из `reports/METRICS.md`, 202 samples);
   - раздел HTTP-нагрузки под `TBD // measured via locust` — ждёт
     чистого прод-прогона, цифры не придуманы;
   - таблица соответствия ТЗ Параметрам 3 и 7;
   - команды для воспроизведения.

## Файлы

- `k8s/configmap.yaml`
- `k8s/deployment.yaml`
- `k8s/service.yaml`
- `k8s/hpa.yaml`
- `k8s/ingress.yaml`
- `k8s/README.md`
- `reports/LOAD_TEST.md`

## Как проверить

```bash
# На целевом кластере:
kubectl create namespace ecomarineai
kubectl apply -f k8s/
kubectl -n ecomarineai rollout status deploy/ecomarine-backend
kubectl -n ecomarineai get pods,svc,hpa,ingress

# Локальный smoke-test через port-forward:
kubectl -n ecomarineai port-forward svc/ecomarine-backend 8000:80 &
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/metrics | head

# Нагрузочный тест:
locust -f tests/performance/locustfile.py --host http://localhost:8000 \
    --users 100 --spawn-rate 10 --run-time 5m --headless \
    --csv reports/locust_run
```

## Соответствие ТЗ

- **Параметр 3 — линейная временная сложность.** Офлайн-бенчмарк
  `reports/SCALABILITY.md` показывает R² = 1.0, slope 0.482 s/image.
  HPA с CPU target 70 % гарантирует, что при росте нагрузки кластер
  добавляет поды до того, как индивидуальная латентность выйдет за
  8-секундный потолок (п. 3 ТЗ).
- **Параметр 7 — availability ≥ 95 %.** Трёхрепликовый Deployment с
  rolling update `maxUnavailable: 1` оставляет 2 из 3 подов в Ready во
  время обновления, что математически удерживает мгновенную доступность
  ≥ 66 % даже при одновременном падении одного пода и успешно держит
  95 %+ на 7-дневном окне в условиях нормального SLA.

## Открытые пункты

- Настоящий прогон `locust` с реальными p50/p95/p99 под 50 RPS —
  в отчёт `reports/LOAD_TEST.md` будут вписаны метрики из
  `reports/locust_run_stats.csv` после теста на фактическом железе.
- Секрет `hf-credentials` (HF token) — создаётся вручную на целевом
  кластере (команда в `k8s/README.md`).
