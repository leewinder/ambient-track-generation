#!/usr/bin/env python3
""" Common utilities for Stable Diffusion XL operations """

import torch
from diffusers import DiffusionPipeline


def get_device() -> str:
    """ Select the best available compute device: MPS (Apple), CUDA (NVIDIA), or CPU fallback """
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) for acceleration")
        return "mps"
    if torch.cuda.is_available():
        print("Using CUDA for acceleration")
        return "cuda"
    print("Using CPU (no GPU acceleration available)")
    return "cpu"


def get_optimal_dtype(device: str) -> torch.dtype:
    """ Get the optimal data type for the given device """
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def optimize_pipeline(pipeline: DiffusionPipeline, device: str) -> None:
    """ Apply standard memory and compute optimizations to a diffusion pipeline """
    # Enable CPU offloading for CUDA devices to save VRAM
    if device == "cuda":
        if hasattr(pipeline, "enable_model_cpu_offload"):
            pipeline.enable_model_cpu_offload()

    # Enable attention slicing to reduce memory usage
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()


def create_generator(device: str, seed: int) -> torch.Generator:
    """ Create a seeded random generator for the specified device """
    return torch.Generator(device=device).manual_seed(seed)
