# import cupy as cp
from scipy.ndimage import median_filter
import numpy as np
    

class MedianFilterNaive:
    def __init__(self, input_array):
        # Convert input array to CuPy array for GPU acceleration
        self.input_array_gpu = input_array
        self.output_array_gpu = np.empty_like(input_array)
        self.mask_gpu = np.empty_like(input_array, dtype=np.bool_)
        self.input_shape = input_array.shape

    def compute_median(self, threshold, window_size = (5, 50)):
        # Ensure the window size is legal
        if (window_size[0] < 3 or window_size[1] < 16 or
            window_size[0] > 32 or window_size[1] > 200):
            raise ValueError("Window size is out of the allowed range.")

        # Compute the median using the specified window size on the GPU cached array
        median_filter(self.input_array_gpu, size=window_size, output=self.output_array_gpu)
        # Add THD to output array
        median_with_thd = self.output_array_gpu + threshold
        # Create the mask by identifying pixels in the original input array that exceed the thresholded median
        self.mask_gpu =  self.input_array_gpu > median_with_thd

    def cupy_to_np(self):
        return self.mask_gpu
    # return cp.asnumpy(self.mask)