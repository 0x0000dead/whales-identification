import pytest
import torch

from whales_identify.model import ArcMarginProduct, CetaceanIdentificationModel, GeM


class TestGeM:
    def test_output_shape(self):
        gem = GeM(p=3, eps=1e-6)
        x = torch.rand(2, 64, 8, 8)
        out = gem(x)
        assert out.shape == (2, 64, 1, 1)

    def test_learnable_param(self):
        gem = GeM(p=3)
        params = list(gem.parameters())
        assert len(params) == 1
        assert params[0].shape == (1,)

    def test_positive_output(self):
        gem = GeM()
        x = torch.rand(1, 32, 4, 4) + 0.01
        out = gem(x)
        assert (out > 0).all()


class TestArcMarginProduct:
    def test_output_shape(self):
        arc = ArcMarginProduct(in_features=512, out_features=100, s=30.0, m=0.5)
        x = torch.randn(4, 512)
        labels = torch.tensor([0, 1, 2, 3])
        out = arc(x, labels)
        assert out.shape == (4, 100)

    def test_different_margins(self):
        arc1 = ArcMarginProduct(64, 10, s=30.0, m=0.3)
        arc2 = ArcMarginProduct(64, 10, s=30.0, m=0.7)
        x = torch.randn(2, 64)
        labels = torch.tensor([0, 1])
        out1 = arc1(x, labels)
        out2 = arc2(x, labels)
        assert out1.shape == out2.shape
        assert not torch.allclose(out1, out2)

    def test_weight_initialization(self):
        arc = ArcMarginProduct(128, 50)
        assert arc.weight.shape == (50, 128)
        assert arc.weight.requires_grad


class TestCetaceanIdentificationModel:
    @pytest.fixture
    def small_model(self):
        return CetaceanIdentificationModel(
            model_name="efficientnet_b0",
            embedding_size=64,
            num_classes=10,
            s=30.0,
            m=0.5,
            ls_eps=0.0,
            easy_margin=False,
        )

    def test_forward_shape(self, small_model):
        images = torch.randn(2, 3, 224, 224)
        labels = torch.tensor([0, 1])
        out = small_model(images, labels)
        assert out.shape == (2, 10)

    def test_model_has_pooling(self, small_model):
        assert hasattr(small_model, "pooling")
        assert isinstance(small_model.pooling, GeM)

    def test_model_has_embedding(self, small_model):
        assert hasattr(small_model, "embedding")
        assert small_model.embedding.out_features == 64

    def test_model_has_arcface(self, small_model):
        assert hasattr(small_model, "fc")
        assert isinstance(small_model.fc, ArcMarginProduct)


def test_legacy_alias_deprecation_warning():
    """The old HappyWhaleModel name still resolves but warns."""
    import warnings

    from whales_identify.model import HappyWhaleModel

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = HappyWhaleModel(
            model_name="efficientnet_b0",
            embedding_size=32,
            num_classes=5,
            s=30.0,
            m=0.5,
            ls_eps=0.0,
            easy_margin=False,
        )
        deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deprecation, "HappyWhaleModel must emit a DeprecationWarning"
        assert isinstance(m, CetaceanIdentificationModel)
