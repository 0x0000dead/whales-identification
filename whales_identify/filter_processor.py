class FilterProcessor:
    """
    Класс для фильтрации данных в датасете, который удаляет записи с невалидными метками и, при необходимости, шумовые пакеты.

    Attributes:
        valid_labels (list): Список допустимых меток (например, типов китов), которые считаются валидными.
        remove_noise (bool): Флаг, указывающий, нужно ли удалять шумовые данные (по умолчанию True).
    """

    def __init__(self, valid_labels, remove_noise=True):
        """
        Инициализация класса фильтрации данных.

        Args:
            valid_labels (list): Список валидных меток, которые должны остаться в датасете.
            remove_noise (bool): Флаг, указывающий, следует ли удалять шумовые пакеты (по умолчанию True).
        """
        self.valid_labels = valid_labels
        self.remove_noise = remove_noise

    def filter_data(self, dataset):
        """
        Фильтрует данные в датасете, удаляя изображения с невалидными метками.
        Если параметр remove_noise установлен в True, будут удаляться шумовые пакеты (если они определены).

        Args:
            dataset (WhaleDataset): Исходный датасет с изображениями и метками.

        Returns:
            WhaleDataset: Отфильтрованный датасет, содержащий только валидные данные.
        """
        filtered_data = {}
        for img_file, label in dataset.labels.items():
            if label in self.valid_labels:
                filtered_data[img_file] = label

        dataset.labels = filtered_data
        return dataset
