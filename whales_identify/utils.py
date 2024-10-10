import numpy as np
import os
import random
import torch

def set_seed(seed=42):
    """
    Устанавливает начальное значение генераторов случайных чисел для обеспечения воспроизводимости.

    Параметры:
        seed (int): Значение семени, по умолчанию 42. Это значение будет использоваться
                    для инициализации всех генераторов случайных чисел.

    Устанавливает семена для:
        - numpy
        - стандартной библиотеки random
        - PyTorch (для CPU и GPU)
        - Ограничивает возможности ускоренной работы cudnn
        - Устанавливает переменную окружения PYTHONHASHSEED для управления хэшированием
          в Python и гарантирует воспроизводимость при использовании словарей.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
