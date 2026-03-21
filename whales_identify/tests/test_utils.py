import os
import random

import numpy as np
import torch

from whales_identify.utils import set_seed


class TestSetSeed:
    def test_numpy_reproducibility(self):
        set_seed(42)
        a = np.random.rand(10)
        set_seed(42)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_random_reproducibility(self):
        set_seed(42)
        a = [random.random() for _ in range(10)]
        set_seed(42)
        b = [random.random() for _ in range(10)]
        assert a == b

    def test_torch_reproducibility(self):
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        assert torch.equal(a, b)

    def test_pythonhashseed_set(self):
        set_seed(123)
        assert os.environ.get("PYTHONHASHSEED") == "123"

    def test_different_seeds_different_output(self):
        set_seed(1)
        a = np.random.rand(10)
        set_seed(2)
        b = np.random.rand(10)
        assert not np.array_equal(a, b)


class TestFilterProcessor:
    def test_filter_keeps_valid_labels(self):
        from whales_identify.filter_processor import FilterProcessor

        processor = FilterProcessor(valid_labels=[0, 1, 2])

        class FakeDataset:
            def __init__(self):
                self.labels = {"a.jpg": 0, "b.jpg": 1, "c.jpg": 5}

        ds = FakeDataset()
        filtered = processor.filter_data(ds)
        assert "a.jpg" in filtered.labels
        assert "b.jpg" in filtered.labels
        assert "c.jpg" not in filtered.labels

    def test_filter_empty_dataset(self):
        from whales_identify.filter_processor import FilterProcessor

        processor = FilterProcessor(valid_labels=[0])

        class FakeDataset:
            def __init__(self):
                self.labels = {}

        ds = FakeDataset()
        filtered = processor.filter_data(ds)
        assert len(filtered.labels) == 0
