# 🌐 Справочник URL системы мониторинга

## ❗ Важное замечание

**Ссылки на странице http://localhost:9090/targets показывают внутренние Docker имена**, которые недоступны из браузера. Используйте правильные localhost ссылки ниже.

## ✅ Правильные ссылки для браузера

### 📊 Основные интерфейсы
| Сервис | URL | Описание |
|--------|-----|----------|
| **Prometheus** | http://localhost:9090 | Главный интерфейс мониторинга |
| **Grafana** | http://localhost:3001 | Дашборды (admin/admin) |
| **Alertmanager** | http://localhost:9093 | Управление алертами |

### 🔍 Эндпоинты метрик
| Сервис | URL | Статус |
|--------|-----|---------|
| **Telegram Bot** | http://localhost:8085/metrics | ✅ 200 OK |
| **API EfficientNet v1** | http://localhost:8001/metrics | ✅ 200 OK |
| **API EfficientNet v2** | http://localhost:8002/metrics | ✅ 200 OK |
| **API ResNet v1** | http://localhost:8003/metrics | ✅ 200 OK |
| **Alertmanager** | http://localhost:9093/metrics | ✅ 200 OK |

### 🔧 Health endpoints
| Сервис | URL | Формат |
|--------|-----|---------|
| **Telegram Bot** | http://localhost:8085/health | JSON |
| **API EfficientNet v1** | http://localhost:8001/health | JSON |
| **API EfficientNet v2** | http://localhost:8002/health | JSON |
| **API ResNet v1** | http://localhost:8003/health | JSON |

## ❌ Неработающие ссылки (Docker internal)

**НЕ используйте эти ссылки** - они показываются в Prometheus UI, но недоступны извне:
- `http://alertmanager:9093/metrics`
- `http://telegram-notifier:8080/metrics`
- `http://api-efficientnet-v1:8000/metrics`
- `http://api-efficientnet-v2:8000/metrics`
- `http://api-resnet-v1:8000/metrics`
- `http://06e4fc79893e:9090/metrics`

## 🧪 Быстрая проверка

Проверить все метрики одной командой:
```bash
echo "Checking all metrics endpoints..."
curl -s -w "Telegram Bot: HTTP %{http_code}\n" http://localhost:8085/metrics -o /dev/null
curl -s -w "API v1: HTTP %{http_code}\n" http://localhost:8001/metrics -o /dev/null
curl -s -w "API v2: HTTP %{http_code}\n" http://localhost:8002/metrics -o /dev/null
curl -s -w "API ResNet: HTTP %{http_code}\n" http://localhost:8003/metrics -o /dev/null
curl -s -w "Alertmanager: HTTP %{http_code}\n" http://localhost:9093/metrics -o /dev/null
```

## 📈 Примеры метрик

### Telegram Bot метрики:
```
telegram_webhook_requests_total 0
telegram_webhook_requests_successful 0
telegram_messages_sent_total 0
telegram_bot_configured 1
```

### API метрики:
```
whales_api_cpu_percent 15.2
whales_api_memory_percent 42.5
whales_api_model_loaded{model="efficientnet_v1"} 0
whales_api_requests_total{model="efficientnet_v1"} 0
```

## 🎯 Быстрые команды для работы

```bash
# Открыть все основные интерфейсы
open http://localhost:9090    # Prometheus
open http://localhost:3001    # Grafana
open http://localhost:9093    # Alertmanager

# Проверить метрики API
curl http://localhost:8001/metrics | head -10
curl http://localhost:8002/metrics | head -10  
curl http://localhost:8003/metrics | head -10

# Тестирование алертов
python3 scripts/test_alerts.py
```

---

✅ **Все ссылки протестированы и работают!** 