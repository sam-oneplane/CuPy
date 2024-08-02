
import numpy as np
from src.median_filter import MedianFilter

def main():
    # Example input data
    input_array = 10*np.random.rand(20, 200).astype(np.float32)

    # Create a cache object
    filter = MedianFilter(input_array)

    # Apply the median filter with threshold using the cached data
    threshold = 0.5
    filter.compute_median((3,16))
    filter.compute_mask_with_threshold(threshold)

    print(filter.cupy_to_np())

if __name__ == "__main__":
    main()