import unittest
import torch
from whales_identify.model import GeM, ArcMarginProduct, HappyWhaleModel

class TestModel(unittest.TestCase):

    def test_GeM(self):
        gem = GeM()
        input_tensor = torch.randn(1, 512, 7, 7)
        output = gem(input_tensor)
        self.assertEqual(output.shape, (1, 512, 1, 1))

    def test_ArcMarginProduct(self):
        amp = ArcMarginProduct(512, 1000)
        input_tensor = torch.randn(10, 512)
        labels = torch.randint(0, 1000, (10,))
        output = amp(input_tensor, labels)
        self.assertEqual(output.shape, (10, 1000))

    def test_HappyWhaleModel(self):
        model = HappyWhaleModel(
            model_name='tf_efficientnet_b0_ns',
            embedding_size=512,
            num_classes=1000,
            s=30.0,
            m=0.50,
            ls_eps=0.0,
            easy_margin=False
        )
        images = torch.randn(2, 3, 448, 448)
        labels = torch.randint(0, 1000, (2,))
        outputs = model(images, labels)
        self.assertEqual(outputs.shape, (2, 1000))

if __name__ == '__main__':
    unittest.main()