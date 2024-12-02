import unittest
from whales_identify.filter_processor import FilterProcessor
from whales_identify.dataset import WhaleDataset

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