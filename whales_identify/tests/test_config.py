import torch

from whales_identify.config import CONFIG


class TestConfig:
    def test_required_keys_exist(self):
        required = [
            "seed",
            "epochs",
            "img_size",
            "model_name",
            "num_classes",
            "embedding_size",
            "train_batch_size",
            "valid_batch_size",
            "learning_rate",
            "scheduler",
            "min_lr",
            "T_max",
            "weight_decay",
            "device",
            "s",
            "m",
            "ls_eps",
            "easy_margin",
        ]
        for key in required:
            assert key in CONFIG, f"Missing config key: {key}"

    def test_num_classes(self):
        assert CONFIG["num_classes"] == 15587

    def test_img_size(self):
        assert CONFIG["img_size"] == 448

    def test_device_is_torch_device(self):
        assert isinstance(CONFIG["device"], torch.device)

    def test_seed_is_set(self):
        assert isinstance(CONFIG["seed"], int)
        assert CONFIG["seed"] > 0

    def test_learning_rate_range(self):
        assert 0 < CONFIG["learning_rate"] < 1

    def test_batch_sizes_positive(self):
        assert CONFIG["train_batch_size"] > 0
        assert CONFIG["valid_batch_size"] > 0
