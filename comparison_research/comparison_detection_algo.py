import time
import numpy as np
import pandas as pd
import cv2
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from transformers import ViTForImageClassification, ViTFeatureExtractor


# Загрузка моделей
def load_model(model_name):
    if model_name == "EfficientNet-B0":
        model = EfficientNet.from_pretrained('efficientnet-b0')
    elif model_name == "ViT":
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        return model, feature_extractor
    else:
        raise ValueError("Неизвестная модель")
    return model


# Предобработка изображений
def preprocess_image(image_path, input_size, model_name):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_size, input_size))

    if model_name == "EfficientNet-B0":
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return preprocess(image).unsqueeze(0)
    elif model_name == "ViT":
        return image


def compare_algorithms(image_path, models):
    results = []
    for model_name, config in models.items():
        print(f"Тестирование модели: {model_name}")
        model, feature_extractor = None, None

        if model_name == "ViT":
            model, feature_extractor = load_model(model_name)
            inputs = feature_extractor(images=image_path, return_tensors="pt")
        else:
            model = load_model(model_name)
            inputs = preprocess_image(image_path, config["input_size"], model_name)

        model.eval()
        start_time = time.time()

        with torch.no_grad():
            if model_name == "ViT":
                outputs = model(**inputs)
            else:
                outputs = model(inputs)

        inference_time = time.time() - start_time
        results.append({
            "Model": model_name,
            "Inference Time (s)": round(inference_time, 4),
            "Accuracy (simulated)": np.random.uniform(0.85, 0.95)  # Симуляция точности для примера
        })
    return pd.DataFrame(results)

# Конфигурация моделей
models = {
    "EfficientNet-B0": {"input_size": 224},
    "ViT": {"input_size": 224}
}

# Тестовое изображение
image_path = "test_image.jpg"

# Сравнение
df_results = compare_algorithms(image_path, models)

# Вывод результатов
print(df_results)
