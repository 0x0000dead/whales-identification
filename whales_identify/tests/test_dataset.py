import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from whales_identify.dataset import WhaleDataset, augmentation_data_transforms


class TestWhaleDataset:
    @pytest.fixture
    def sample_dataset(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        labels = {}
        for i in range(5):
            name = f"whale_{i}.jpg"
            img = Image.fromarray(
                np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            )
            img.save(img_dir / name)
            labels[name] = i

        return WhaleDataset(str(img_dir), labels)

    def test_length(self, sample_dataset):
        assert len(sample_dataset) == 5

    def test_getitem_returns_dict(self, sample_dataset):
        item = sample_dataset[0]
        assert isinstance(item, dict)
        assert "image" in item
        assert "label" in item

    def test_getitem_label_type(self, sample_dataset):
        item = sample_dataset[0]
        assert isinstance(item["label"], int)

    def test_missing_label_returns_negative(self, tmp_path):
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        img = Image.fromarray(np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8))
        img.save(img_dir / "unknown.jpg")

        ds = WhaleDataset(str(img_dir), labels={})
        item = ds[0]
        assert item["label"] == -1

    def test_with_transforms(self, tmp_path):
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(img_dir / "test.jpg")

        transforms = augmentation_data_transforms()
        ds = WhaleDataset(str(img_dir), {"test.jpg": 0}, transforms["valid"])
        item = ds[0]
        assert item["image"].shape[0] == 3  # channels first


class TestAugmentationTransforms:
    def test_returns_train_and_valid(self):
        transforms = augmentation_data_transforms()
        assert "train" in transforms
        assert "valid" in transforms

    def test_train_transform_exists(self):
        transforms = augmentation_data_transforms()
        assert transforms["train"] is not None

    def test_valid_transform_exists(self):
        transforms = augmentation_data_transforms()
        assert transforms["valid"] is not None
