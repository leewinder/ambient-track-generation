#!/usr/bin/env python3
""" Common utilities for Stable Diffusion XL operations """

import torch
from diffusers import DiffusionPipeline
from . import logging_utils

# Module-level logger
logger = logging_utils.get_logger(__name__)


# Device constants
class Devices:
    """ Constants for supported device types """
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


# Valid device types for SDXL operations
VALID_DEVICES = {Devices.CUDA, Devices.MPS, Devices.CPU}


def get_device() -> str:
    """ Select the best available compute device: MPS (Apple), CUDA (NVIDIA), or CPU fallback """
    if torch.backends.mps.is_available():
        logger.info("Using MPS (Apple Silicon) for acceleration")
        return Devices.MPS
    if torch.cuda.is_available():
        logger.info("Using CUDA for acceleration")
        return Devices.CUDA
    logger.info("Using CPU (no GPU acceleration available)")
    return Devices.CPU


def get_optimal_dtype(device: str) -> torch.dtype:
    """ Get the optimal data type for the given device """
    if device not in VALID_DEVICES:
        raise ValueError(f"Invalid device '{device}'. Must be one of: {', '.join(VALID_DEVICES)}")

    if device == Devices.CUDA:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def optimize_pipeline(pipeline: DiffusionPipeline, device: str) -> None:
    """ Apply standard memory and compute optimizations to a diffusion pipeline """
    if device not in VALID_DEVICES:
        raise ValueError(f"Invalid device '{device}'. Must be one of: {', '.join(VALID_DEVICES)}")

    logger.debug("Optimizing pipeline for device: %s", device)

    # Enable CPU offloading for CUDA devices to save VRAM
    if device == Devices.CUDA:
        if hasattr(pipeline, "enable_model_cpu_offload"):
            pipeline.enable_model_cpu_offload()
            logger.debug("Enabled CPU offloading for CUDA device")

    # Enable attention slicing to reduce memory usage
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()
        logger.debug("Enabled attention slicing for memory optimization")


def create_generator(device: str, seed: int) -> torch.Generator:
    """ Create a seeded random generator for the specified device """
    if device not in VALID_DEVICES:
        raise ValueError(f"Invalid device '{device}'. Must be one of: {', '.join(VALID_DEVICES)}")

    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got: {seed}")

    logger.debug("Creating generator for device %s with seed %d", device, seed)
    return torch.Generator(device=device).manual_seed(seed)
