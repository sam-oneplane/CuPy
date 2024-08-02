#### `setup.py`
from setuptools import setup, find_packages

setup(
    name="median_filter",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "cupy-cuda11x",
        "numpy",
    ],
)