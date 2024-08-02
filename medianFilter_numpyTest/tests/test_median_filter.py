import unittest
import numpy as np
from src.median_filter import MedianFilter
from src.median_filter_naive import MedianFilterNaive

class TestComputeMedianWithThreshold(unittest.TestCase):

    def setUp(self):
        self.test_array = 10*np.random.rand(50, 500).astype(np.float32)
        self.m_filter = MedianFilterNaive(self.test_array)
        self.threshold = 0.5


    def test_default_window(self):
        input_array = 10*np.random.rand(20, 200).astype(np.float32)
        filter = MedianFilter(input_array)
        threshold = 0.5
        filter.compute_median_with_downsample(threshold)
        mask = filter.cupy_to_np()
        self.assertEqual(mask.shape, input_array.shape)
        self.assertTrue(np.issubdtype(mask.dtype, np.bool_))
        

    def test_default_window(self):

        input_array = np.full((20, 80), 5.0)
        filter = MedianFilter(input_array)
        naive_filter = MedianFilterNaive(input_array)
        threshold = 0.5
        filter.compute_median_with_downsample((3,16), threshold)
        mask0 = filter.cupy_to_np()
        naive_filter.compute_median((3,16), threshold)
        mask1 = naive_filter.cupy_to_np()
        equal = np.allclose(mask0, mask1)
        self.assertEqual(equal, True)

    
    def test_invalid_window_size(self):
        
        with self.assertRaises(ValueError):
            self.m_filter.compute_median((2, 50), self.threshold)
        with self.assertRaises(ValueError):
            self.m_filter.compute_median((5, 15), self.threshold)
        with self.assertRaises(ValueError):
            self.m_filter.compute_median((33, 50), self.threshold)
        with self.assertRaises(ValueError):
            self.m_filter.compute_median((5, 201), self.threshold)


if __name__ == '__main__':
    unittest.main()
