import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """
    Класс Generalized Mean Pooling (GeM) для агрегирования признаков в сверточной нейронной сети.

    Attributes:
        p (nn.Parameter): Параметр степени для усреднения, обучаемый параметр.
        eps (float): Малое значение для предотвращения деления на ноль при использовании операции clamp.
    """

    def __init__(self, p=3, eps=1e-6):
        """
        Инициализация модуля GeM.

        Args:
            p (float, optional): Начальное значение для параметра степени. По умолчанию 3.
            eps (float, optional): Малое значение для предотвращения деления на ноль. По умолчанию 1e-6.
        """
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        """
        Прямой проход через слой GeM.

        Args:
            x (torch.Tensor): Входной тензор с признаками.

        Returns:
            torch.Tensor: Пуленый (усреднённый) тензор.
        """
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1. / self.p)


class ArcMarginProduct(nn.Module):
    """
    Класс для вычисления ArcFace функции потерь с ArcMarginProduct, используемой для задачи классификации с высокой
    межклассовой разделимостью.

    Attributes:
        in_features (int): Размерность входного пространства признаков.
        out_features (int): Количество выходных классов (размерность выходного пространства).
        s (float): Коэффициент масштабирования для нормализации.
        m (float): Дуга отступа для увеличения разделимости классов.
        easy_margin (bool): Легкая версия margin, если включена.
        ls_eps (float): Параметр метки сглаживания для уменьшения переобучения.
        weight (nn.Parameter): Обучаемые веса классификатора.
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        """
        Инициализация модуля ArcMarginProduct.

        Args:
            in_features (int): Размерность входных признаков.
            out_features (int): Количество классов.
            s (float, optional): Коэффициент масштабирования. По умолчанию 30.0.
            m (float, optional): Дуга отступа для улучшения разделимости. По умолчанию 0.50.
            easy_margin (bool, optional): Использовать ли легкий margin. По умолчанию False.
            ls_eps (float, optional): Параметр сглаживания меток. По умолчанию 0.0.
        """
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.ls_eps = ls_eps
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        """
        Прямой проход через ArcMarginProduct.

        Args:
            input (torch.Tensor): Входные признаки.
            label (torch.Tensor): Метки классов для каждого примера.

        Returns:
            torch.Tensor: Выходные логиты с учетом margin.
        """
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * torch.cos(self.m) - sine * torch.sin(self.m)
        output = (torch.where(cosine > torch.cos(torch.pi - self.m), phi,
                              cosine - torch.sin(torch.pi - self.m) * self.m) * self.s)
        one_hot = torch.zeros(cosine.size(), device=label.device).scatter_(1, label.view(-1, 1).long(), 1)
        return (one_hot * output + (1.0 - one_hot) * cosine) * self.s


class HappyWhaleModel(nn.Module):
    """
   Класс модели для задачи классификации китов с использованием ArcFace и Generalized Mean Pooling.

   Attributes:
       model (nn.Module): Предобученная модель из библиотеки timm, используемая как основа.
       pooling (GeM): Слой GeM для пуллинга признаков.
       embedding (nn.Linear): Линейный слой для уменьшения размерности признаков до embedding_size.
       fc (ArcMarginProduct): Слой для применения ArcMarginProduct с метками.
   """

    def __init__(self, model_name, embedding_size, num_classes, s, m, ls_eps, easy_margin):
        """
        Инициализация модели HappyWhale.

        Args:
            model_name (str): Название предобученной модели из библиотеки timm.
            embedding_size (int): Размерность пространства embedding.
            num_classes (int): Количество классов для классификации.
            s (float): Коэффициент масштабирования для ArcMarginProduct.
            m (float): Margin отступ для ArcMarginProduct.
            ls_eps (float): Параметр сглаживания меток.
            easy_margin (bool): Флаг для включения легкого margin.
        """
        super(HappyWhaleModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.embedding = nn.Linear(in_features, embedding_size)
        self.fc = ArcMarginProduct(embedding_size, num_classes, s=s, m=m, ls_eps=ls_eps, easy_margin=easy_margin)

    def forward(self, images, labels):
        """
        Прямой проход через модель.

        Args:
            images (torch.Tensor): Входные изображения.
            labels (torch.Tensor): Метки классов.

        Returns:
            torch.Tensor: Логиты для классификации с использованием ArcMarginProduct.
        """
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        embedding = self.embedding(pooled_features)
        return self.fc(embedding, labels)
