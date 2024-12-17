import torch
import time
import cv2

# Загрузка предобученной модели YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

img_path = "data/sample.png"
image = cv2.imread(img_path)
assert image is not None, "Ошибка: изображение не загружено. Проверьте путь."

start_time = time.time()

results = model(img_path)

end_time = time.time()
execution_time = end_time - start_time

detection_results = results.pandas().xyxy[0]
mean_confidence = detection_results['confidence'].mean()

print(f"Время исполнения: {execution_time:.4f} секунд")
print(f"Средняя точность (confidence): {mean_confidence:.4f}")

results.show()
results.save()

results_df = results.pandas().xyxy[0]
results_df.to_csv("yolo_results.csv", index=False)
print("Результаты сохранены в 'yolo_results.csv'")
