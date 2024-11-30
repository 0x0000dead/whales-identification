import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class WhaleDataset(Dataset):
    """
    Класс для создания кастомного датасета, который загружает изображения китов из указанной директории
    и возвращает соответствующие метки. Может применять трансформации к изображениям.

    Attributes:
        img_dir (str): Путь к директории с изображениями.
        labels (dict): Словарь, где ключи — имена файлов изображений, а значения — метки (типы китов).
        transform (callable, optional): Трансформации, которые будут применяться к изображениям (например, аугментации).
        image_files (list): Список файлов изображений в директории img_dir.
    """

    def __init__(self, img_dir, labels, transform=None):
        """
        Инициализация датасета.

        Args:
            img_dir (str): Путь к директории с изображениями.
            labels (dict): Словарь, содержащий метки для каждого изображения (ключ - имя файла).
            transform (callable, optional): Трансформации для изображений. По умолчанию None.
        """
        self.img_dir = img_dir
        self.labels = labels
        self.transform = transform
        self.image_files = os.listdir(img_dir)

    def __len__(self):
        """
        Возвращает количество изображений в датасете.

        Returns:
            int: Количество изображений.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Возвращает изображение и соответствующую метку по индексу.

        Args:
            idx (int): Индекс изображения.

        Returns:
            tuple: (image, label), где image - изображение после трансформации, label - метка для этого изображения.
        """
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")  # открываем изображение
        # метка или -1 если нет метки
        label = self.labels.get(self.image_files[idx], -1)

        # Применяем аугментации через Albumentations
        if self.transform:
            image = self.transform(image=np.array(image))['image']

        return image, label  # Возвращаем изображение и метку


def augmentation_data_transforms() -> dict:
    """
    Функция для определения аугментаций данных для обучения и валидации.

    Возвращает:
        dict: Словарь с аугментациями для тренировочных и валидационных данных.
    """
    data_transforms = {"train": A.Compose([A.Resize(CONFIG['img_size'], CONFIG['img_size']),
                                           A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=60,
                                                              p=0.5),
                                           A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2,
                                                                val_shift_limit=0.2, p=0.5),
                                           A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                                                      contrast_limit=(-0.1, 0.1), p=0.5),
                                           A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                                                       max_pixel_value=255.0, p=1.0), ToTensorV2()],
                                          p=1.),

                       "valid": A.Compose([A.Resize(CONFIG['img_size'], CONFIG['img_size']),
                                           A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                                                       max_pixel_value=255.0, p=1.0),
                                           ToTensorV2()], p=1.)}
    return data_transforms


if __name__ == "__main__":
    from albumentations.pytorch import ToTensorV2
    import albumentations as A
    from whales_identify.config import CONFIG

    # Пример аугментаций
    data_transforms = augmentation_data_transforms()
    # Пример создания тренировочного и валидационного датасетов
    # train_labels и valid_labels - это словари, где ключи - имена файлов, а значения - метки
    # train_dataset = WhaleDataset(img_dir='path/to/train_images', labels=train_labels,
    #                              transform=data_transforms['train'])
    #
    # valid_dataset = WhaleDataset(img_dir='path/to/valid_images', labels=valid_labels,
    #                              transform=data_transforms['valid'])
