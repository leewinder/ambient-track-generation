""" Setup script for the SDXL utilities package """
from setuptools import setup, find_packages

setup(
    name="pipeline-sdxl",
    version="0.1.0",
    description="SDXL utilities for Stable Diffusion XL functionality",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "pipeline-utilities>=0.1.0",
        "torch==2.8.0",
        "torchvision==0.23.0",
        "torchaudio==2.8.0",
        "diffusers==0.30.0",
    ],
)
