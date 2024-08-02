import cupy as cp
from cupyx.scipy.ndimage import median_filter, zoom
import numpy as np
    

class MedianFilter:
    def __init__(self, input_array):
        # Convert input array to CuPy array for GPU acceleration
        self.input_array_gpu = cp.array(input_array)
        self.output_array_gpu = cp.empty_like(input_array)
        self.mask_gpu = cp.empty_like(input_array, dtype=np.bool_)
        self.input_shape = input_array.shape

    def compute_median(self, window_size = (5, 50)):
        # Ensure the window size is legal
        if (window_size[0] < 3 or window_size[1] < 16 or
            window_size[0] > 32 or window_size[1] > 200):
            raise ValueError("Window size is out of the allowed range.")

        # Compute the median using the specified window size on the GPU cached array
        median_filter(self.input_array_gpu, size=window_size, output=self.output_array_gpu)

    def compute_median_with_downsample(self, window_size = (5, 50), zoom_value = 2.0):
        # Ensure the window size is legal
        if (window_size[0] < 3 or window_size[1] < 16 or
            window_size[0] > 32 or window_size[1] > 200):
            raise ValueError("Window size is out of the allowed range.")
        
        downsampled_win_size = (cp.int_(window_size[0] // zoom_value), cp.int_(window_size[1] // zoom_value))
        downsampled_input_size = (cp.int_(self.input_shape[0] // zoom_value), cp.int_(self.input_shape[1] // zoom_value))
        downsampled_input = cp.empty(shape=downsampled_input_size)
        downsampled_output = cp.empty(shape=downsampled_input_size)
        # downsample input array by a factor of zoom (default = 2.0)
        zoom(input=self.input_array_gpu, zoom=1/zoom_value, output=downsampled_input, order=3, mode='reflect')
        # Compute the median on a downsampled input_array
        median_filter(downsampled_input, size=downsampled_win_size, output=downsampled_output)
        # upsample back to original shape
        zoom(input=downsampled_output, zoom=zoom_value, output=self.output_array_gpu, order=3, mode='reflect')

    def compute_mask_with_threshold(self, threshold):
        # Add THD to output array
        median_with_thd = self.output_array_gpu + threshold
        # Create the mask by identifying pixels in the original input array that exceed the thresholded median
        self.mask_gpu =  self.input_array_gpu > median_with_thd

    def cupy_to_np(self):
        return cp.asnumpy(self.mask)
