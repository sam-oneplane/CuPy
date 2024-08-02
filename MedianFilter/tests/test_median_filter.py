import unittest
import numpy as np
from src.median_filter import MedianFilter

class TestComputeMedianWithThreshold(unittest.TestCase):

    def setUp(self):
        self.test_array = 10*np.random.rand(50, 500).astype(np.float32)
        self.m_filter = MedianFilter(self.test_array)

    def test_default_window(self):
        input_array = 10*np.random.rand(20, 200).astype(np.float32)
        filter = MedianFilter(input_array)
        threshold = 0.5
        filter.compute_median_with_downsample()
        filter.compute_mask_with_threshold(threshold)
        mask = filter.cupy_to_np()
        self.assertEqual(mask.shape, self.input_array.shape)
        self.assertTrue(np.issubdtype(mask.dtype, np.bool_))
        

    def test_default_window(self):

        input_array = np.full((20, 80), 5.0)
        filter = MedianFilter(input_array)
        threshold = 0.5
        filter.compute_median((3,16))
        filter.compute_mask_with_threshold(threshold)
        mask0 = filter.cupy_to_np()
        filter.compute_median_with_downsample()
        filter.compute_mask_with_threshold(threshold)
        mask1 = filter.cupy_to_np()
        equal = np.allclose(mask0, mask1)
        self.assertEqual(equal, True)

    
    def test_invalid_window_size(self):
        
        with self.assertRaises(ValueError):
            self.m_filter.compute_median((2, 50))
        with self.assertRaises(ValueError):
            self.m_filter.compute_median((5, 15))
        with self.assertRaises(ValueError):
            self.m_filter.compute_median((33, 50))
        with self.assertRaises(ValueError):
            self.m_filter.compute_median((5, 201))


if __name__ == '__main__':
    unittest.main()
