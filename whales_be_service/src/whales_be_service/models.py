import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Any
import logging
import time
from pathlib import Path

from .config import get_model_config, API_CONFIG


class GeM(nn.Module):
    """Generalized Mean Pooling для агрегирования признаков"""
    
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return nn.functional.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 
            (x.size(-2), x.size(-1))
        ).pow(1. / self.p)


class HappyWhaleModel(nn.Module):
    """Модель для идентификации китов"""
    
    def __init__(self, model_name: str, embedding_size: int, num_classes: int):
        super(HappyWhaleModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        
        # Определение количества входных признаков в зависимости от архитектуры
        if hasattr(self.model, 'classifier'):
            # EfficientNet и подобные
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        elif hasattr(self.model, 'fc'):
            # ResNet и подобные
            in_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif hasattr(self.model, 'head'):
            # Vision Transformer и подобные
            in_features = self.model.head.in_features
            self.model.head = nn.Identity()
        else:
            raise ValueError(f"Неизвестная архитектура модели для {model_name}")
            
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.embedding = nn.Linear(in_features, embedding_size)
        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, images):
        """Прямой проход для инференса"""
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        embeddings = self.embedding(pooled_features)
        logits = self.classifier(embeddings)
        return embeddings, logits


class ModelManager:
    """Менеджер для управления различными моделями"""
    
    def __init__(self):
        self.models: Dict[str, HappyWhaleModel] = {}
        self.transforms: Dict[str, A.Compose] = {}
        self.device = torch.device(API_CONFIG["device"])
        self.logger = logging.getLogger(__name__)
        
    def load_model(self, model_name: str) -> None:
        """Загрузка модели из файла"""
        if model_name in self.models:
            self.logger.info(f"Model {model_name} already loaded")
            return
            
        config = get_model_config(model_name)
        
        # Создание модели
        model = HappyWhaleModel(
            model_name=config["model_name"],
            embedding_size=config["embedding_size"],
            num_classes=config["num_classes"]
        )
        
        # Загрузка весов если файл существует
        model_path = Path(config["model_path"])
        if model_path.exists():
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                self.logger.info(f"Loaded weights from {model_path}")
            except Exception as e:
                self.logger.warning(f"Could not load weights from {model_path}: {e}")
                self.logger.info("Using pretrained weights only")
        else:
            self.logger.warning(f"Model file {model_path} not found, using pretrained weights only")
        
        model.to(self.device)
        model.eval()
        
        # Создание трансформаций для данной модели
        img_size = config["img_size"]
        transforms = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2()
        ])
        
        self.models[model_name] = model
        self.transforms[model_name] = transforms
        self.logger.info(f"Model {model_name} loaded successfully")
    
    def preprocess_image(self, image: np.ndarray, model_name: str) -> torch.Tensor:
        """Предобработка изображения для конкретной модели"""
        if model_name not in self.transforms:
            raise ValueError(f"Model {model_name} not loaded")
        
        # Конвертация в RGB если необходимо
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Применение трансформаций
        transformed = self.transforms[model_name](image=image)
        tensor = transformed["image"].unsqueeze(0)  # Добавляем batch dimension
        return tensor.to(self.device)
    
    def predict(self, image: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Выполнение предсказания"""
        if model_name not in self.models:
            self.load_model(model_name)
        
        start_time = time.time()
        
        # Предобработка
        tensor = self.preprocess_image(image, model_name)
        preprocessing_time = time.time() - start_time
        
        # Инференс
        inference_start = time.time()
        with torch.no_grad():
            embeddings, logits = self.models[model_name](tensor)
            probabilities = torch.softmax(logits, dim=1)
            
        inference_time = time.time() - inference_start
        
        # Постобработка
        top5_probs, top5_indices = torch.topk(probabilities, 5, dim=1)
        
        result = {
            "model_name": model_name,
            "model_version": get_model_config(model_name)["version"],
            "predictions": [
                {
                    "whale_id": f"W-{idx.item():05d}",
                    "confidence": prob.item(),
                    "class_index": idx.item()
                }
                for prob, idx in zip(top5_probs[0], top5_indices[0])
            ],
            "embedding": embeddings[0].cpu().numpy().tolist(),
            "metrics": {
                "preprocessing_time": preprocessing_time,
                "inference_time": inference_time,
                "total_time": time.time() - start_time,
                "model_name": model_name
            }
        }
        
        return result
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Получение информации о модели"""
        config = get_model_config(model_name)
        is_loaded = model_name in self.models
        
        info = {
            "name": model_name,
            "version": config["version"],
            "description": config["description"],
            "is_loaded": is_loaded,
            "config": config
        }
        
        if is_loaded:
            model = self.models[model_name]
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            info.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "device": str(next(model.parameters()).device)
            })
        
        return info


# Глобальный менеджер моделей
model_manager = ModelManager() 