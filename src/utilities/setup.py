""" Setup script for the common utilities package """
from setuptools import setup, find_packages

setup(
    name="pipeline-utilities",
    version="0.1.0",
    description="Common utilities for AI image pipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "pydantic>=2.0.0",
        "torch==2.8.0",
        "torchvision==0.23.0",
        "torchaudio==2.8.0",
        "diffusers==0.30.0",
    ],
)
