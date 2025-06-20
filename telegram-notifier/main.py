#!/usr/bin/env python3
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Telegram Notifier for Whale Models")

# Конфигурация
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "webhook_secret")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    logger.error("TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID должны быть установлены!")

# Метрики для мониторинга
webhook_requests_total = 0
webhook_requests_successful = 0
webhook_requests_failed = 0
telegram_messages_sent = 0
telegram_messages_failed = 0

def send_telegram_message(message: str) -> bool:
    """Отправка сообщения в Telegram"""
    global telegram_messages_sent, telegram_messages_failed
    
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram не настроен")
        telegram_messages_failed += 1
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("Сообщение успешно отправлено в Telegram")
        telegram_messages_sent += 1
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка отправки в Telegram: {e}")
        telegram_messages_failed += 1
        return False

def format_alert_message(alert: Dict[str, Any]) -> str:
    """Форматирование алерта для Telegram"""
    status = alert.get("status", "unknown")
    
    # Эмодзи для разных типов алертов
    if status == "firing":
        status_emoji = "🚨" if alert.get("labels", {}).get("severity") == "critical" else "⚠️"
    else:
        status_emoji = "✅"
    
    # Основная информация
    alert_name = alert.get("labels", {}).get("alertname", "Unknown")
    model = alert.get("labels", {}).get("model", "Unknown")
    severity = alert.get("labels", {}).get("severity", "info")
    
    # Аннотации
    summary = alert.get("annotations", {}).get("summary", "")
    description = alert.get("annotations", {}).get("description", "")
    
    # Время
    starts_at = alert.get("startsAt", "")
    if starts_at:
        try:
            dt = datetime.fromisoformat(starts_at.replace('Z', '+00:00'))
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except:
            time_str = starts_at
    else:
        time_str = "Unknown"
    
    # Формирование сообщения
    message = f"{status_emoji} <b>{alert_name}</b>\n\n"
    
    if model != "Unknown":
        message += f"🤖 <b>Модель:</b> {model}\n"
    
    message += f"📊 <b>Серьезность:</b> {severity.upper()}\n"
    message += f"📅 <b>Время:</b> {time_str}\n"
    message += f"🔄 <b>Статус:</b> {status.upper()}\n\n"
    
    if summary:
        message += f"📝 <b>Краткое описание:</b>\n{summary}\n\n"
    
    if description:
        message += f"📋 <b>Детали:</b>\n{description}\n\n"
    
    # Дополнительные лейблы
    labels = alert.get("labels", {})
    if labels:
        filtered_labels = {k: v for k, v in labels.items() 
                          if k not in ["alertname", "model", "severity", "job", "instance"]}
        if filtered_labels:
            message += f"🏷️ <b>Дополнительная информация:</b>\n"
            for key, value in filtered_labels.items():
                message += f"  • {key}: {value}\n"
    
    return message

@app.post("/webhook")
async def receive_alert(request: Request):
    """Получение алертов от Alertmanager"""
    global webhook_requests_total, webhook_requests_successful, webhook_requests_failed
    
    webhook_requests_total += 1
    
    try:
        # Проверка заголовков
        content_type = request.headers.get("content-type", "")
        if not content_type.startswith("application/json"):
            webhook_requests_failed += 1
            return JSONResponse(
                status_code=400,
                content={"error": "Content-Type должен быть application/json"}
            )
        
        # Получение данных
        body = await request.json()
        
        # Логирование
        logger.info(f"Получен webhook: {json.dumps(body, indent=2)}")
        
        # Обработка алертов
        alerts = body.get("alerts", [])
        if not alerts:
            logger.warning("Нет алертов в webhook")
            webhook_requests_successful += 1
            return JSONResponse(content={"status": "no alerts"})
        
        # Отправка каждого алерта
        sent_count = 0
        for alert in alerts:
            message = format_alert_message(alert)
            if send_telegram_message(message):
                sent_count += 1
        
        webhook_requests_successful += 1
        return JSONResponse(content={
            "status": "success",
            "processed_alerts": len(alerts),
            "sent_messages": sent_count
        })
        
    except json.JSONDecodeError:
        logger.error("Ошибка парсинга JSON")
        webhook_requests_failed += 1
        return JSONResponse(
            status_code=400,
            content={"error": "Неверный формат JSON"}
        )
    except Exception as e:
        logger.error(f"Ошибка обработки webhook: {e}")
        webhook_requests_failed += 1
        return JSONResponse(
            status_code=500,
            content={"error": f"Внутренняя ошибка: {str(e)}"}
        )

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "telegram_configured": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
    }

@app.get("/metrics")
async def get_metrics():
    """Метрики для Prometheus"""
    from fastapi.responses import PlainTextResponse
    
    metrics = [
        f"telegram_webhook_requests_total {webhook_requests_total}",
        f"telegram_webhook_requests_successful {webhook_requests_successful}",
        f"telegram_webhook_requests_failed {webhook_requests_failed}",
        f"telegram_messages_sent_total {telegram_messages_sent}",
        f"telegram_messages_failed_total {telegram_messages_failed}",
        f"telegram_bot_configured {1 if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID else 0}",
    ]
    
    return PlainTextResponse(
        content="\n".join(metrics) + "\n",
        media_type="text/plain"
    )

@app.post("/test")
async def test_notification():
    """Тестовое уведомление"""
    test_message = """🧪 <b>Тестовое уведомление</b>

🤖 <b>Модель:</b> test-model
📊 <b>Серьезность:</b> INFO
📅 <b>Время:</b> """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC") + """
🔄 <b>Статус:</b> TESTING

📝 <b>Краткое описание:</b>
Это тестовое сообщение для проверки работы Telegram бота

📋 <b>Детали:</b>
Система уведомлений для мониторинга моделей идентификации китов работает корректно"""

    success = send_telegram_message(test_message)
    
    return JSONResponse(content={
        "status": "success" if success else "error",
        "message": "Тестовое уведомление отправлено" if success else "Ошибка отправки"
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080) 