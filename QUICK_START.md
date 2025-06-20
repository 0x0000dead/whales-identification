# 🐋 Быстрый запуск системы мониторинга китов

## 📱 1. Настройка Telegram бота (2 минуты)

### Создайте бота:
1. Найдите `@BotFather` в Telegram
2. Отправьте `/newbot`
3. Следуйте инструкциям
4. Сохраните токен бота

### Получите Chat ID:
1. Добавьте бота в чат
2. Напишите боту сообщение
3. Откройте: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
4. Найдите `chat.id` в ответе

### Создайте файл `.env`:
```bash
TELEGRAM_BOT_TOKEN=1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk
TELEGRAM_CHAT_ID=-1001234567890
```

## 🚀 2. Запуск системы (1 команда)

```bash
docker-compose up -d
```

## 🔍 3. Проверка работы (5 минут)

### Доступные сервисы:
- **API EfficientNet v1:** http://localhost:8001
- **API EfficientNet v2:** http://localhost:8002  
- **API ResNet v1:** http://localhost:8003
- **Grafana:** http://localhost:3001 (admin/admin)
- **Prometheus:** http://localhost:9090
- **Alertmanager:** http://localhost:9093

### Тест API:
```bash
curl -X POST "http://localhost:8001/predict" \
     -F "file=@test_image.jpg"
```

### Тест Telegram:
```bash
curl -X POST http://localhost:8085/test
```

### Проверка метрик:
```bash
curl http://localhost:8001/metrics
```

## 🧪 4. Тестирование алертов

```bash
# Запуск автоматического тестирования
python scripts/test_alerts.py
```

Этот скрипт:
- ✅ Проверит все сервисы
- ✅ Отправит тестовое уведомление в Telegram
- ✅ Создаст нагрузку для срабатывания алертов
- ✅ Покажет статистику работы

## 📊 5. Мониторинг в Grafana

1. Откройте http://localhost:3001
2. Логин: `admin`, Пароль: `admin`
3. Найдите дашборд "Whales Models Performance"
4. Наблюдайте метрики в реальном времени

## 🎯 Ожидаемые результаты

### Telegram уведомления:
- 🧪 Тестовое сообщение при `/test`
- 🚨 Алерт при времени инференса > 5 сек (ResNet)
- ⚠️ Алерт при ошибках > 10%
- 💻 Алерт при высокой нагрузке CPU/памяти

### Grafana дашборды:
- ⏱️ Время инференса по моделям
- 📊 RPS и количество запросов
- ❌ Коэффициент ошибок
- 🔄 Статус загрузки моделей
- 💾 Использование ресурсов

### Prometheus метрики:
```
whales_api_requests_total{model="efficientnet_v1"} 150
whales_api_inference_time_seconds{model="resnet_v1"} 19.2
whales_api_error_rate{model="efficientnet_v2"} 0.05
```

## ❓ Проблемы?

**Telegram не работает?**
```bash
# Проверьте логи
docker-compose logs telegram-notifier

# Проверьте переменные
docker exec whales-telegram-notifier env | grep TELEGRAM
```

**Метрики не собираются?**
```bash
# Проверьте эндпоинты
curl http://localhost:8001/metrics
curl http://localhost:8002/metrics
curl http://localhost:8003/metrics
```

**Grafana не показывает данные?**
```bash
# Проверьте Prometheus
curl http://localhost:9090/api/v1/query?query=up
```

## 📋 Полная документация

Подробное руководство: `HOMEWORK_MONITORING_GUIDE.md`

---

**Готово! 🎉 Система мониторинга запущена и готова к работе.** 