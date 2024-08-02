import numpy as np


class HistogramMedian:
    def __init__(self, input_array, num_bins=256):
        # Convert input array to CuPy array for GPU acceleration
        self.input_array_gpu = input_array # cp.array(input_array)
        self.num_bins = num_bins
        self.max_intensity = np.max(self.input_array_gpu)
        self.min_intensity = np.min(self.input_array_gpu)
        self.hist_range = self.max_intensity - self.min_intensity
        self.bin_width = self.hist_range / self.num_bins
        self.bin_edges = np.linspace(self.min_intensity, self.max_intensity, self.num_bins + 1)
        self.mask_gpu = np.empty_like(self.input_array_gpu, dtype=np.bool_)
        
    def compute_histogram(self, window):
        histogram, _ = np.histogram(window, bins=self.bin_edges)
        return histogram

    def find_median_from_histogram(self, histogram, num_elements):
        cumulative_hist = np.cumsum(histogram)
        median_bin = np.searchsorted(cumulative_hist, num_elements // 2)
        median_value = self.min_intensity + median_bin * self.bin_width
        return median_value

    def compute_median_with_threshold(self, window_size, threshold):
        # Ensure the window size is legal
        if (window_size[0] < 3 or window_size[1] < 16 or 
            window_size[0] > 32 or window_size[1] > 200):
            raise ValueError("Window size is out of the allowed range.")
        
        pad_height = window_size[0] // 2
        pad_width = window_size[1] // 2
        padded_array = np.pad(self.input_array_gpu, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')
        

        for i in range(self.input_array_gpu.shape[0]):
            for j in range(self.input_array_gpu.shape[1]):
                window = padded_array[i:i + window_size[0], j:j + window_size[1]]
                histogram = self.compute_histogram(window)
                median = self.find_median_from_histogram(histogram, window_size[0] * window_size[1])
                median_with_threshold = median + threshold
                self.mask_gpu[i, j] = self.input_array_gpu[i, j] > median_with_threshold
        
        
    def cupy_to_np(self):
        return self.mask_gpu
        #return cp.asnumpy(self.mask)