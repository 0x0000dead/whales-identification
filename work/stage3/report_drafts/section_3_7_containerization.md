# §3.7 — Контейнеризация и файл запуска (черновик для НТО)

_Ответственный: Тарасов А.А._
_Статус: реализовано_
_Связанные пункты ТЗ: Параметр 3 (линейная временная сложность),
Параметр 7 (availability ≥ 95 %)._

## Что сделано

1. **Production Docker Compose** — добавлен `docker-compose.prod.yml`:
   - `restart: always` вместо `unless-stopped` (выдерживает перезагрузку
     хоста и OOM-kill);
   - лимиты ресурсов `cpus: 2.0` / `memory: 4G` на backend, согласованные
     с Kubernetes манифестом (`k8s/deployment.yaml`), и 0.5 CPU / 512 MiB
     на frontend;
   - именованные volume `ecomarine-logs`, `ecomarine-reports`,
     `ecomarine-nginx-logs` — переживают пересоздание контейнеров,
     готовы к подключению лог-шиппера;
   - profile `smoke`, запускающий лёгкий curl-контейнер
     `curlimages/curl:8.10.1`, который последовательно вызывает `/health`,
     `/metrics`, `/v1/drift-stats` (см. `scripts/smoke_test.sh`
     для полноценного варианта).

2. **Скрипт запуска для Linux / macOS** — `scripts/start.sh`:
   - проверяет наличие и доступность Docker-демона;
   - автоматически запускает `scripts/download_models.sh`, если весов
     нет в `whales_be_service/src/whales_be_service/models/`
     (идемпотентно — повторный запуск пропускает шаг);
   - запускает `docker compose up -d --remove-orphans` на выбранном
     compose-файле (`dev` или `prod`);
   - опрашивает `http://localhost:8000/health` в цикле до таймаута
     `HEALTH_TIMEOUT=180 s`;
   - печатает URL сервисов и команду остановки;
   - возвращает различающиеся коды выхода для каждой стадии отказа
     (1 — Docker, 2 — download, 3 — compose, 4 — health).
   - Синтаксис проверен `bash -n`.

3. **Скрипт запуска для Windows** — `scripts/start.bat`:
   - проверка Docker Desktop и наличия `curl.exe`;
   - вызов `bash scripts/download_models.sh` через Git Bash / WSL2;
   - `docker compose -f ... up -d` и опрос `/health` через `curl.exe`
     с максимум 90 × 2 с = 180 с ожидания.

4. **Smoke-test профиль** — `docker compose -f docker-compose.prod.yml
   --profile smoke run --rm smoke` проверяет работоспособность стека
   в CI без необходимости устанавливать Python или Locust на раннере.

## Файлы

- `docker-compose.prod.yml`
- `scripts/start.sh` (исполняемый)
- `scripts/start.bat`

## Как проверить

```bash
# Dev-запуск:
./scripts/start.sh

# Production-запуск:
./scripts/start.sh prod

# Smoke-тест через compose profile:
docker compose -f docker-compose.prod.yml --profile smoke run --rm smoke

# Остановка:
docker compose -f docker-compose.prod.yml down
```

Ожидаемые результаты: `/health` отдаёт HTTP 200 за ≤ 180 с после старта
(включая скачивание ~500 MiB весов), `/metrics` экспортирует счётчики
`availability_percent`, `latency_avg_ms`, `requests_total`. При повторном
запуске `start.sh` веса не перекачиваются, время выхода в Ready < 30 с.

## Соответствие ТЗ

- **Параметр 3 (линейная временная сложность)** — production compose
  использует тот же backend образ, что и k8s-манифест, поэтому результаты
  `reports/SCALABILITY.md` (slope 0.482 s/image, R² = 1.0) переносятся
  напрямую.
- **Параметр 7 (availability ≥ 95 %)** — `restart: always`,
  healthcheck-и на уровне compose и HEALTHCHECK в Dockerfile, лимиты
  ресурсов и volume для логов — минимальный контур для долгоживущего
  развёртывания; метрика отслеживается через
  `/metrics availability_percent`.
