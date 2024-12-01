import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from whales_identify.config import CONFIG
from whales_identify.dataset import WhaleDataset, augmentation_data_transforms
from whales_identify.model import HappyWhaleModel
from whales_identify.utils import set_seed


def train_one_epoch(model, dataloader, optimizer, device):
    """
    Обучение модели на одной эпохе.

    Args:
        model (torch.nn.Module): Обучаемая модель.
        dataloader (DataLoader): DataLoader с обучающими данными.
        optimizer (torch.optim.Optimizer): Оптимизатор для обновления весов модели.
        device (torch.device): Устройство, на котором выполняется обучение (CPU или GPU).
    """
    model.train()
    for data in dataloader:
        images, labels = data['image'].to(device), data['label'].to(device)
        outputs = model(images, labels)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def run_training(df_train: pd.DataFrame, img_dir: str):
    """
    Запуск процесса обучения модели на всех эпохах.
    Настраивает модель, датасеты, оптимизатор и планировщик обучения.

    Args:
        df_train (pd.DataFrame): Данные для тренировки модели
        img_dir (str): путь к директории с картинками для тренировочных данных ('path/to/train_images')
    """
    set_seed(CONFIG['seed'])
    device = CONFIG['device']

    model = HappyWhaleModel(CONFIG['model_name'], CONFIG['embedding_size'],
                            CONFIG['num_classes'], CONFIG['s'], CONFIG['m'],
                            CONFIG['ls_eps'], CONFIG['easy_margin'])
    model.to(device)

    data_transforms = augmentation_data_transforms()
    train_labels = {row['file_path']: row['individual_id']
                    for index, row in df_train.iterrows()}

    train_dataset = WhaleDataset(img_dir=img_dir,
                                 labels=train_labels,
                                 transform=data_transforms["train"])
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['train_batch_size'])

    optimizer = Adam(model.parameters(
    ), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = CosineAnnealingLR(
        optimizer, T_max=CONFIG['T_max'], eta_min=CONFIG['min_lr'])

    # Цикл по эпохам для обучения.
    for epoch in range(CONFIG['epochs']):
        train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()
