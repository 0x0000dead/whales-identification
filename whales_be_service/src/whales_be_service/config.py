import os
from typing import Dict, Any

# Конфигурация для различных моделей
MODEL_CONFIGS = {
    "efficientnet_v1": {
        "model_name": "tf_efficientnet_b0_ns",
        "model_path": "/app/models/efficientnet_v1.pt",
        "embedding_size": 512,
        "num_classes": 15587,
        "img_size": 448,
        "version": "1.0",
        "description": "EfficientNet B0 trained on whale dataset"
    },
    "efficientnet_v2": {
        "model_name": "tf_efficientnet_b2_ns", 
        "model_path": "/app/models/efficientnet_v2.pt",
        "embedding_size": 512,
        "num_classes": 15587,
        "img_size": 512,
        "version": "2.0",
        "description": "EfficientNet B2 with improved accuracy"
    },
    "resnet_v1": {
        "model_name": "resnet101",
        "model_path": "/app/models/resnet101.pth",
        "embedding_size": 512,
        "num_classes": 15587,
        "img_size": 448,
        "version": "1.0",
        "description": "ResNet101 baseline model"
    }
}

# Основная конфигурация API
API_CONFIG = {
    "default_model": os.getenv("DEFAULT_MODEL", "efficientnet_v1"),
    "max_file_size": int(os.getenv("MAX_FILE_SIZE", "10485760")),  # 10MB
    "batch_size": int(os.getenv("BATCH_SIZE", "8")),
    "device": os.getenv("DEVICE", "cpu"),
    "log_level": os.getenv("LOG_LEVEL", "INFO"),
    "enable_metrics": os.getenv("ENABLE_METRICS", "true").lower() == "true"
}

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Получить конфигурацию модели по имени"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name]

def get_available_models() -> Dict[str, str]:
    """Получить список доступных моделей"""
    return {name: config["description"] for name, config in MODEL_CONFIGS.items()} 