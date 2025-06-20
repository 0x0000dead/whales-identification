#!/usr/bin/env python3
"""
Скрипт для тестирования системы мониторинга и алертов
Генерирует различные сценарии нагрузки для срабатывания алертов
"""

import asyncio
import aiohttp
import time
import random
import logging
from typing import List, Dict, Any
import json
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация
API_ENDPOINTS = [
    "http://localhost:8001",  # EfficientNet v1
    "http://localhost:8002",  # EfficientNet v2  
    "http://localhost:8003",  # ResNet v1
]

TELEGRAM_NOTIFIER = "http://localhost:8085"

class AlertTester:
    def __init__(self):
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()

    async def check_services_health(self) -> Dict[str, bool]:
        """Проверка здоровья всех сервисов"""
        logger.info("🔍 Проверка состояния сервисов...")
        
        results = {}
        
        # Проверка API сервисов
        for endpoint in API_ENDPOINTS:
            try:
                async with self.session.get(f"{endpoint}/health", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        results[endpoint] = True
                        logger.info(f"✅ {endpoint} - OK")
                    else:
                        results[endpoint] = False
                        logger.warning(f"⚠️ {endpoint} - HTTP {response.status}")
            except Exception as e:
                results[endpoint] = False
                logger.error(f"❌ {endpoint} - {str(e)}")
        
        # Проверка Telegram notifier
        try:
            async with self.session.get(f"{TELEGRAM_NOTIFIER}/health", timeout=5) as response:
                if response.status == 200:
                    results[TELEGRAM_NOTIFIER] = True
                    logger.info(f"✅ Telegram notifier - OK")
                else:
                    results[TELEGRAM_NOTIFIER] = False
                    logger.warning(f"⚠️ Telegram notifier - HTTP {response.status}")
        except Exception as e:
            results[TELEGRAM_NOTIFIER] = False
            logger.error(f"❌ Telegram notifier - {str(e)}")
            
        return results

    async def test_telegram_notification(self) -> bool:
        """Тестирование Telegram уведомлений"""
        logger.info("📱 Тестирование Telegram уведомлений...")
        
        try:
            async with self.session.post(f"{TELEGRAM_NOTIFIER}/test", timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ Тестовое уведомление отправлено: {data}")
                    return True
                else:
                    logger.error(f"❌ Ошибка отправки: HTTP {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ Ошибка тестирования Telegram: {str(e)}")
            return False

    async def generate_high_load(self, endpoint: str, duration: int = 60) -> Dict[str, Any]:
        """Генерация высокой нагрузки для тестирования алертов"""
        logger.info(f"🚀 Генерация нагрузки на {endpoint} в течение {duration} секунд...")
        
        # Создаем тестовое изображение (пустой файл)
        test_image = b"fake_image_data" * 1000  # Имитация изображения
        
        start_time = time.time()
        requests_sent = 0
        successful_requests = 0
        errors = 0
        
        # Список задач для параллельного выполнения
        tasks = []
        
        async def send_request():
            nonlocal requests_sent, successful_requests, errors
            
            try:
                # FormData для файла
                data = aiohttp.FormData()
                data.add_field('file', test_image, filename='test.jpg', content_type='image/jpeg')
                
                async with self.session.post(
                    f"{endpoint}/predict", 
                    data=data,
                    timeout=30
                ) as response:
                    requests_sent += 1
                    
                    if response.status == 200:
                        successful_requests += 1
                    else:
                        errors += 1
                        
            except Exception as e:
                requests_sent += 1
                errors += 1
                logger.debug(f"Ошибка запроса: {str(e)}")
        
        # Генерируем нагрузку
        while time.time() - start_time < duration:
            # Создаем пакет из 10 параллельных запросов
            for _ in range(10):
                tasks.append(asyncio.create_task(send_request()))
            
            # Ждем завершения части запросов
            if len(tasks) >= 50:
                await asyncio.gather(*tasks[:25], return_exceptions=True)
                tasks = tasks[25:]
            
            await asyncio.sleep(0.1)  # Небольшая пауза
        
        # Ждем завершения оставшихся запросов
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        duration_actual = end_time - start_time
        
        result = {
            "endpoint": endpoint,
            "duration": duration_actual,
            "requests_sent": requests_sent,
            "successful_requests": successful_requests,
            "errors": errors,
            "error_rate": errors / requests_sent if requests_sent > 0 else 0,
            "rps": requests_sent / duration_actual if duration_actual > 0 else 0
        }
        
        logger.info(f"📊 Результаты нагрузки для {endpoint}:")
        logger.info(f"   Отправлено запросов: {requests_sent}")
        logger.info(f"   Успешных: {successful_requests}")
        logger.info(f"   Ошибок: {errors}")
        logger.info(f"   Коэффициент ошибок: {result['error_rate']:.2%}")
        logger.info(f"   RPS: {result['rps']:.2f}")
        
        return result

    async def check_prometheus_metrics(self) -> Dict[str, Any]:
        """Проверка метрик в Prometheus"""
        logger.info("📈 Проверка метрик Prometheus...")
        
        prometheus_url = "http://localhost:9090"
        metrics_queries = [
            "whales_api_requests_total",
            "whales_api_error_rate", 
            "whales_api_inference_time_seconds",
            "whales_api_model_loaded"
        ]
        
        results = {}
        
        for query in metrics_queries:
            try:
                async with self.session.get(
                    f"{prometheus_url}/api/v1/query",
                    params={"query": query},
                    timeout=10
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results[query] = data.get("data", {}).get("result", [])
                        logger.info(f"✅ Метрика {query}: {len(results[query])} значений")
                    else:
                        results[query] = None
                        logger.warning(f"⚠️ Метрика {query}: HTTP {response.status}")
            except Exception as e:
                results[query] = None
                logger.error(f"❌ Метрика {query}: {str(e)}")
        
        return results

    async def trigger_memory_alert(self, endpoint: str) -> bool:
        """Попытка вызвать алерт по использованию памяти"""
        logger.info(f"🧠 Попытка вызвать алерт по памяти для {endpoint}...")
        
        # Генерируем много параллельных запросов для увеличения использования памяти
        tasks = []
        test_image = b"large_fake_image_data" * 10000  # Большое "изображение"
        
        for _ in range(100):  # 100 параллельных запросов
            async def memory_intensive_request():
                try:
                    data = aiohttp.FormData()
                    data.add_field('file', test_image, filename='large_test.jpg', content_type='image/jpeg')
                    
                    async with self.session.post(
                        f"{endpoint}/predict",
                        data=data,
                        timeout=60
                    ) as response:
                        return response.status == 200
                except:
                    return False
            
            tasks.append(asyncio.create_task(memory_intensive_request()))
        
        # Выполняем все запросы параллельно
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful = sum(1 for r in results if r is True)
        
        logger.info(f"📊 Выполнено {successful}/{len(tasks)} запросов для нагрузки памяти")
        return successful > 0

async def main():
    """Основная функция тестирования"""
    logger.info("🐋 Запуск тестирования системы мониторинга китов")
    
    async with AlertTester() as tester:
        # 1. Проверка здоровья сервисов
        health_status = await tester.check_services_health()
        healthy_services = [k for k, v in health_status.items() if v]
        
        if not healthy_services:
            logger.error("❌ Нет доступных сервисов для тестирования")
            return
        
        # 2. Тестирование Telegram уведомлений
        telegram_works = await tester.test_telegram_notification()
        if not telegram_works:
            logger.warning("⚠️ Telegram уведомления не работают")
        
        # 3. Проверка метрик Prometheus
        metrics = await tester.check_prometheus_metrics()
        
        # 4. Тестирование различных сценариев алертов
        test_scenarios = [
            {
                "name": "Тест быстрой модели (EfficientNet v1)",
                "endpoint": "http://localhost:8001",
                "duration": 30,
                "expected_alerts": ["Возможно высокая нагрузка"]
            },
            {
                "name": "Тест сбалансированной модели (EfficientNet v2)",
                "endpoint": "http://localhost:8002", 
                "duration": 30,
                "expected_alerts": ["Умеренная нагрузка"]
            },
            {
                "name": "Тест медленной модели (ResNet)",
                "endpoint": "http://localhost:8003",
                "duration": 45,
                "expected_alerts": ["HighInferenceTime - время > 5 сек"]
            }
        ]
        
        load_results = []
        
        for scenario in test_scenarios:
            if scenario["endpoint"] in healthy_services:
                logger.info(f"\n🧪 {scenario['name']}")
                result = await tester.generate_high_load(
                    scenario["endpoint"], 
                    scenario["duration"]
                )
                load_results.append(result)
                
                # Пауза между тестами
                logger.info("⏳ Пауза 15 секунд между тестами...")
                await asyncio.sleep(15)
        
        # 5. Попытка вызвать алерт по памяти
        if healthy_services:
            await tester.trigger_memory_alert(healthy_services[0])
        
        # 6. Итоговый отчет
        logger.info("\n📋 ИТОГОВЫЙ ОТЧЕТ ТЕСТИРОВАНИЯ")
        logger.info("=" * 50)
        
        logger.info(f"🔍 Проверено сервисов: {len(health_status)}")
        logger.info(f"✅ Здоровых сервисов: {len(healthy_services)}")
        logger.info(f"📱 Telegram работает: {'✅' if telegram_works else '❌'}")
        logger.info(f"📈 Prometheus метрики: {len([k for k, v in metrics.items() if v])}/{len(metrics)}")
        
        logger.info("\n📊 Результаты нагрузочного тестирования:")
        for result in load_results:
            logger.info(f"  {result['endpoint']}:")
            logger.info(f"    RPS: {result['rps']:.2f}")
            logger.info(f"    Ошибок: {result['error_rate']:.2%}")
            logger.info(f"    Запросов: {result['requests_sent']}")
        
        logger.info("\n🎯 Ожидаемые алерты:")
        logger.info("  - HighInferenceTime (если ResNet > 5 сек)")
        logger.info("  - HighErrorRate (если ошибок > 10%)")
        logger.info("  - HighCPUUsage (если нагрузка > 80%)")
        logger.info("  - HighMemoryUsage (если память > 85%)")
        
        logger.info("\n📱 Проверьте Telegram на наличие уведомлений!")
        logger.info("🌐 Grafana: http://localhost:3001")
        logger.info("📊 Prometheus: http://localhost:9090")
        logger.info("🤖 Alertmanager: http://localhost:9093")

if __name__ == "__main__":
    asyncio.run(main()) 