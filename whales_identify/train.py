import os
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from whales_identify.config import CONFIG
from whales_identify.dataset import WhaleDataset, augmentation_data_transforms
from whales_identify.model import HappyWhaleModel
from whales_identify.utils import set_seed


def train_one_epoch(model, dataloader, optimizer, device, epoch, total_epochs):
    """
    Обучение модели на одной эпохе.

    Args:
        model (torch.nn.Module): Обучаемая модель.
        dataloader (DataLoader): DataLoader с обучающими данными.
        optimizer (torch.optim.Optimizer): Оптимизатор для обновления весов модели.
        device (torch.device): Устройство, на котором выполняется обучение (CPU или GPU).
        epoch (int): Текущий номер эпохи.
        total_epochs (int): Общее количество эпох.
    """
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    running_loss = 0.0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{total_epochs}")

    for batch_idx, data in progress_bar:
        images, labels = data['image'].to(device), data['label'].to(device)
        optimizer.zero_grad()
        outputs = model(images, labels)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 10 == 0:
            progress_bar.set_postfix({'loss': running_loss / (batch_idx + 1)})

    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{total_epochs} - Loss: {epoch_loss:.4f}")
    return epoch_loss


def save_checkpoint(state, checkpoint_dir, epoch):
    """
    Сохранение состояния модели и оптимизатора.

    Args:
        state (dict): Состояние для сохранения (модель, оптимизатор и т.д.).
        checkpoint_dir (str): Директория для сохранения чекпойнтов.
        epoch (int): Текущий номер эпохи.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
    torch.save(state, checkpoint_path)
    print(f"Сохранен чекпойнт: {checkpoint_path}")


def run_training(df_train: pd.DataFrame, img_dir: str, checkpoint_dir: str):
    """
    Запуск процесса обучения модели на всех эпохах.
    Настраивает модель, датасеты, оптимизатор и планировщик обучения.

    Args:
        df_train (pd.DataFrame): Данные для тренировки модели
        img_dir (str): Путь к директории с картинками для тренировочных данных ('path/to/train_images')
        checkpoint_dir (str): Путь к директории для сохранения чекпойнтов модели
    """
    set_seed(CONFIG['seed'])
    device = CONFIG['device']

    model = HappyWhaleModel(
        CONFIG['model_name'],
        CONFIG['embedding_size'],
        CONFIG['num_classes'],
        CONFIG['s'],
        CONFIG['m'],
        CONFIG['ls_eps'],
        CONFIG['easy_margin']
    )
    model.to(device)

    data_transforms = augmentation_data_transforms()
    train_labels = {row['file_path']: row['individual_id'] for index, row in df_train.iterrows()}

    train_dataset = WhaleDataset(
        img_dir=img_dir,
        labels=train_labels,
        transform=data_transforms["train"]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['train_batch_size'],
        shuffle=True,
        num_workers=4,  # Рекомендуется настроить в соответствии с вашим оборудованием
        pin_memory=True
    )

    optimizer = Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=CONFIG['T_max'],
        eta_min=CONFIG['min_lr']
    )

    best_loss = float('inf')

    # Цикл по эпохам для обучения.
    for epoch in range(CONFIG['epochs']):
        epoch_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, CONFIG['epochs'])
        scheduler.step()

        # Сохранение чекпойнта после каждой эпохи
        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': epoch_loss
        }
        save_checkpoint(checkpoint_state, checkpoint_dir, epoch)

        # Опционально: сохранение лучшей модели по loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Лучшая модель обновлена и сохранена: {best_model_path}")


def main():
    """
    Главная функция для запуска процесса обучения.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train HappyWhaleModel")
    parser.add_argument('--train_csv', type=str, required=True, help="Путь к CSV файлу с тренировочными данными")
    parser.add_argument('--img_dir', type=str, required=True, help="Путь к директории с изображениями для тренировки")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Директория для сохранения чекпойнтов")
    args = parser.parse_args()

    # Загрузка данных
    df_train = pd.read_csv(args.train_csv)

    # Запуск обучения
    run_training(df_train, args.img_dir, args.checkpoint_dir)


if __name__ == '__main__':
    main()