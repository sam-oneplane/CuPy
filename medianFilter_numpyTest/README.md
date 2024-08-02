# Median Filter Project

## Overview

This project implement median filter using CuPy for GPU acceleration. 
The median filter is applied to a 2D input array with a specified window size (50, 5000).
There are 2 functions to implement the median filter defined in 
class MedianFilter.

1. compute_median():
        compute naive median filter using cupyx.scipy.ndimage
2. compute_median_with_downsample():
        attempt to increse speed by using cupy zoom to interpolate and extrapolate both the input
        array and the window size by some factor.


The other function defined in 
class MedianFilter.

1. compute_mask_with_threshold() : 
        add threshold to the computed filter and check if the result is smaller 
        then the input value via mask array.
2. cupy_to_np():
        return traformed mask array from cupy to numpy. 
         

## Installation And Tests

To install the required dependencies, run:

```sh
pip install -r requirements.txt

To run the tests:

```sh
python3 -m unittest discover -s tests

To run the example:

```sh
python3 scripts/run_example.py

