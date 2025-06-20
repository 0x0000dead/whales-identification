from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette import status
from starlette.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
import io
import logging
import time
from typing import Optional, List, Dict, Any
import os

# Временная заглушка для psutil для демонстрации
class MockPsutil:
    @staticmethod
    def cpu_percent():
        return 15.2
    
    @staticmethod
    def virtual_memory():
        class Memory:
            def __init__(self):
                self.percent = 42.5
                self.used = 8589934592  # 8GB
                self.available = 12884901888  # 12GB
        return Memory()

psutil = MockPsutil()

from .models import model_manager
from .config import get_available_models, API_CONFIG, get_model_config

# Настройка логирования
logging.basicConfig(level=getattr(logging, API_CONFIG["log_level"]))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Whales Identification API",
    description="API для идентификации китов с поддержкой множественных моделей",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Метрики для мониторинга
class MetricsCollector:
    def __init__(self):
        self.requests_count = {}
        self.total_inference_time = {}
        self.error_count = {}
    
    def record_request(self, model_name: str, inference_time: float, success: bool = True):
        if model_name not in self.requests_count:
            self.requests_count[model_name] = 0
            self.total_inference_time[model_name] = 0.0
            self.error_count[model_name] = 0
        
        self.requests_count[model_name] += 1
        if success:
            self.total_inference_time[model_name] += inference_time
        else:
            self.error_count[model_name] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        metrics = {}
        for model_name in self.requests_count:
            total_requests = self.requests_count[model_name]
            avg_time = (self.total_inference_time[model_name] / 
                       (total_requests - self.error_count[model_name])) if total_requests > self.error_count[model_name] else 0
            
            metrics[model_name] = {
                "total_requests": total_requests,
                "successful_requests": total_requests - self.error_count[model_name],
                "error_count": self.error_count[model_name],
                "average_inference_time": avg_time,
                "error_rate": self.error_count[model_name] / total_requests if total_requests > 0 else 0
            }
        return metrics

metrics_collector = MetricsCollector()

def validate_image(file: UploadFile) -> np.ndarray:
    """Валидация и конвертация загруженного изображения"""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Поддерживаются только изображения"
        )
    
    if file.size and file.size > API_CONFIG["max_file_size"]:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Файл слишком большой. Максимальный размер: {API_CONFIG['max_file_size']} байт"
        )
    
    try:
        # Чтение и конвертация изображения
        image_bytes = file.file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Конвертация в numpy array
        image_array = np.array(image)
        
        # Проверка размерности
        if len(image_array.shape) not in [2, 3]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Неподдерживаемый формат изображения"
            )
        
        # Конвертация в RGB если нужно
        if len(image_array.shape) == 3 and image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        elif len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        
        return image_array
        
    except Exception as e:
        logger.error(f"Ошибка обработки изображения: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Не удалось обработать изображение"
        )

@app.get("/", summary="Главная страница")
async def root():
    """Информация об API"""
    return {
        "message": "Whales Identification API",
        "version": "2.0.0",
        "available_endpoints": [
            "/models",
            "/models/{model_name}/info",
            "/predict",
            "/predict/{model_name}",
            "/health",
            "/metrics"
        ]
    }

@app.get("/models", summary="Список доступных моделей")
async def get_models():
    """Получить список всех доступных моделей"""
    available_models = get_available_models()
    models_info = {}
    
    for model_name in available_models:
        try:
            models_info[model_name] = model_manager.get_model_info(model_name)
        except Exception as e:
            models_info[model_name] = {
                "name": model_name,
                "error": str(e)
            }
    
    return {
        "available_models": available_models,
        "models_info": models_info,
        "default_model": API_CONFIG["default_model"]
    }

@app.get("/models/{model_name}/info", summary="Информация о модели")
async def get_model_info(model_name: str):
    """Получить подробную информацию о конкретной модели"""
    try:
        return model_manager.get_model_info(model_name)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )

@app.post("/predict", summary="Предсказание с моделью по умолчанию")
async def predict_default(file: UploadFile = File(...)):
    """Выполнить предсказание с моделью по умолчанию"""
    return await predict_with_model(API_CONFIG["default_model"], file)

@app.post("/predict/{model_name}", summary="Предсказание с указанной моделью")
async def predict_with_model(model_name: str, file: UploadFile = File(...)):
    """Выполнить предсказание с указанной моделью"""
    start_time = time.time()
    
    try:
        # Валидация модели
        if model_name not in get_available_models():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Модель '{model_name}' не найдена"
            )
        
        # Валидация и обработка изображения
        image_array = validate_image(file)
        
        # Предсказание
        result = model_manager.predict(image_array, model_name)
        
        # Записываем метрики
        if API_CONFIG["enable_metrics"]:
            metrics_collector.record_request(
                model_name, 
                result["metrics"]["total_time"], 
                success=True
            )
        
        # Добавляем информацию о файле
        result["file_info"] = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": file.size
        }
        
        return JSONResponse(result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка предсказания для модели {model_name}: {str(e)}")
        
        # Записываем ошибку в метрики
        if API_CONFIG["enable_metrics"]:
            metrics_collector.record_request(
                model_name, 
                time.time() - start_time, 
                success=False
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Внутренняя ошибка сервера при выполнении предсказания"
        )

@app.post("/compare", summary="Сравнение предсказаний разных моделей")
async def compare_models(
    file: UploadFile = File(...),
    models: List[str] = Query(None, description="Список моделей для сравнения")
):
    """Сравнить предсказания нескольких моделей на одном изображении"""
    if not models:
        models = list(get_available_models().keys())
    
    # Валидация изображения
    image_array = validate_image(file)
    
    results = {}
    total_start = time.time()
    
    for model_name in models:
        try:
            if model_name not in get_available_models():
                results[model_name] = {"error": f"Модель '{model_name}' не найдена"}
                continue
            
            result = model_manager.predict(image_array, model_name)
            results[model_name] = result
            
            if API_CONFIG["enable_metrics"]:
                metrics_collector.record_request(
                    model_name, 
                    result["metrics"]["total_time"], 
                    success=True
                )
                
        except Exception as e:
            logger.error(f"Ошибка при предсказании модели {model_name}: {str(e)}")
            results[model_name] = {"error": str(e)}
            
            if API_CONFIG["enable_metrics"]:
                metrics_collector.record_request(
                    model_name, 
                    0, 
                    success=False
                )
    
    return {
        "file_info": {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": file.size
        },
        "total_comparison_time": time.time() - total_start,
        "results": results
    }

@app.get("/health", summary="Проверка здоровья сервиса")
async def health_check():
    """Проверка состояния API и загруженных моделей"""
    health_info = {
        "status": "healthy",
        "timestamp": time.time(),
        "api_config": {
            "default_model": API_CONFIG["default_model"],
            "device": API_CONFIG["device"],
            "enable_metrics": API_CONFIG["enable_metrics"]
        },
        "loaded_models": []
    }
    
    for model_name in get_available_models():
        try:
            model_info = model_manager.get_model_info(model_name)
            health_info["loaded_models"].append({
                "name": model_name,
                "is_loaded": model_info["is_loaded"],
                "version": model_info["version"]
            })
        except Exception as e:
            health_info["loaded_models"].append({
                "name": model_name,
                "is_loaded": False,
                "error": str(e)
            })
    
    return health_info

@app.get("/metrics", summary="Метрики работы API в формате Prometheus")
async def get_metrics():
    """Получить метрики использования моделей в формате Prometheus"""
    if not API_CONFIG["enable_metrics"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Метрики отключены"
        )
    
    # Получаем метрики коллектора
    metrics = metrics_collector.get_metrics()
    
    # Системные метрики
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    # Формируем метрики в формате Prometheus
    prometheus_metrics = []
    
    # Метрики по моделям
    for model_name, model_metrics in metrics.items():
        model_label = f'model="{model_name}"'
        
        prometheus_metrics.extend([
            f"whales_api_requests_total{{{model_label}}} {model_metrics['total_requests']}",
            f"whales_api_requests_successful{{{model_label}}} {model_metrics['successful_requests']}",
            f"whales_api_requests_errors{{{model_label}}} {model_metrics['error_count']}",
            f"whales_api_inference_time_seconds{{{model_label}}} {model_metrics['average_inference_time']}",
            f"whales_api_error_rate{{{model_label}}} {model_metrics['error_rate']}",
        ])
    
    # Системные метрики
    prometheus_metrics.extend([
        f"whales_api_cpu_percent {cpu_percent}",
        f"whales_api_memory_percent {memory.percent}",
        f"whales_api_memory_used_bytes {memory.used}",
        f"whales_api_memory_available_bytes {memory.available}",
    ])
    
    # Метрики состояния моделей
    for model_name in get_available_models():
        model_loaded = 1 if model_name in model_manager.models else 0
        prometheus_metrics.append(f'whales_api_model_loaded{{model="{model_name}"}} {model_loaded}')
    
    # Добавляем временную метку
    prometheus_metrics.append(f"whales_api_last_scrape_timestamp {int(time.time())}")
    
    return PlainTextResponse(
        content="\n".join(prometheus_metrics) + "\n",
        media_type="text/plain"
    )

@app.post("/models/{model_name}/load", summary="Принудительная загрузка модели")
async def load_model(model_name: str):
    """Принудительно загрузить модель в память"""
    try:
        if model_name not in get_available_models():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Модель '{model_name}' не найдена"
            )
        
        model_manager.load_model(model_name)
        return {
            "message": f"Модель '{model_name}' успешно загружена",
            "model_info": model_manager.get_model_info(model_name)
        }
        
    except Exception as e:
        logger.error(f"Ошибка загрузки модели {model_name}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось загрузить модель: {str(e)}"
        )
