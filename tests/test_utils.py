import unittest
import torch
import random
import numpy as np
from whales_identify.utils import set_seed

class TestUtils(unittest.TestCase):

    def test_set_seed(self):
        set_seed(42)
        torch_rand1 = torch.randn(1).item()
        np_rand1 = np.random.rand()
        random_rand1 = random.random()

        set_seed(42)
        torch_rand2 = torch.randn(1).item()
        np_rand2 = np.random.rand()
        random_rand2 = random.random()

        self.assertEqual(torch_rand1, torch_rand2)
        self.assertEqual(np_rand1, np_rand2)
        self.assertEqual(random_rand1, random_rand2)

class TestFilterProcessor(unittest.TestCase):

    def test_filter_data(self):
        dataset = WhaleDataset(
            img_dir='dummy_dir',
            labels={'img1.jpg': 'label1', 'img2.jpg': 'invalid_label'}
        )
        processor = FilterProcessor(valid_labels=['label1'])
        filtered_dataset = processor.filter_data(dataset)
        self.assertEqual(len(filtered_dataset.labels), 1)
        self.assertIn('img1.jpg', filtered_dataset.labels)
        self.assertNotIn('img2.jpg', filtered_dataset.labels)

if __name__ == '__main__':
    unittest.main()