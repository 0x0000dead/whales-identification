# Research Notebooks

> ⚠️ **Important**: This directory contains **research and experimentation code**, not production-ready implementations. These notebooks are used for model evaluation, architecture comparison, and experimental analysis.

## Purpose

This `research/notebooks/` directory serves as a sandbox for:

- **Model architecture evaluation** - Comparing different neural network architectures (CNNs, Vision Transformers, etc.)
- **Hyperparameter experimentation** - Testing various training configurations
- **Performance benchmarking** - Measuring accuracy, speed, and resource usage
- **Exploratory data analysis** - Understanding the whale identification dataset

## Relationship to Main Library

| Component                 | Location              | Purpose                                         |
| ------------------------- | --------------------- | ----------------------------------------------- |
| **Research notebooks**    | `research/notebooks/` | Experimentation and evaluation (this directory) |
| **Production ML library** | `whales_identify/`    | Core training and model code for production     |
| **Production API**        | `whales_be_service/`  | FastAPI backend service                         |
| **Demo applications**     | `research/demo-ui/`   | Streamlit demos for visualization               |

**Note**: Code validated in these research notebooks may be promoted to the main `whales_identify/` library after thorough testing and review. However, notebooks themselves should **not be used in production** environments.

## Model Comparison Results

В результате проведенного анализа были получены следующие результаты.

| Критерий                               | CNN (ResNet-54)             | CNN (ResNet-101)             | Metric Learning (EfficientNet-B0) | Metric Learning (EfficientNet-B5)            | ViT-B/16                              | ViT-L/32                              | Swin-T                      |
| -------------------------------------- | --------------------------- | ---------------------------- | --------------------------------- | -------------------------------------------- | ------------------------------------- | ------------------------------------- | --------------------------- |
| **Точность (Precision)**               | 82%                         | 85%                          | 88%                               | 91%                                          | 91%                                   | 93%                                   | 90%                         |
| **Скорость обработки (средняя)**       | ~0.8 секунды                | ~1.2 секунды                 | ~1.0 секунда                      | ~1.8 секунды                                 | ~2.0 секунды                          | ~3.5 секунды                          | ~2.2 секунды                |
| **Масштабируемость**                   | Хорошая, линейная сложность | Средняя, увеличенные ресурсы | Высокая, линейная сложность       | Средняя, ресурсоемкая                        | Средняя, увеличивается с данными      | Низкая, требует значительных ресурсов | Высокая, линейная сложность |
| **Универсальность и адаптивность**     | Средняя                     | Высокая                      | Высокая, устойчива к изменениям   | Очень высокая, устойчива к изменениям        | Очень высокая                         | Очень высокая                         | Высокая                     |
| **Интерфейс и удобство использования** | Простой интерфейс           | Более сложный интерфейс      | Требует настройки эмбеддингов     | Требует более сложной настройки эмбеддингов  | Требует оптимизации для пользователей | Требует высокой оптимизации           | Простой интерфейс           |
| **Интеграция с другими системами**     | Легко интегрируется         | Поддерживает интеграцию      | Совместим с базами данных         | Совместим, но требует дополнительных модулей | Требует модулей для интеграции        | Требует модулей и оптимизации         | Легко интегрируется         |
| **Надежность и стабильность**          | 94% доступности             | 92% доступности              | 95% доступности                   | 93% доступности                              | 93% доступности                       | 90% доступности                       | 94% доступности             |
| **Чувствительность (Sensitivity)**     | 78%                         | 82%                          | 85%                               | 88%                                          | 89%                                   | 91%                                   | 90%                         |
| **Специфичность (Specificity)**        | 88%                         | 90%                          | 92%                               | 94%                                          | 91%                                   | 92%                                   | 91%                         |
| **Полнота (Recall)**                   | 76%                         | 80%                          | 85%                               | 88%                                          | 89%                                   | 91%                                   | 90%                         |
| **F1-мера**                            | 0.79                        | 0.82                         | 0.86                              | 0.89                                         | 0.90                                  | 0.92                                  | 0.91                        |
