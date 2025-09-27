#!/usr/bin/env python3
""" Common utilities for Stable Diffusion XL operations """

import os
import re
from pathlib import Path
from typing import Type, TypeVar

import torch
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline
from pipeline_utilities import logging_utils

# Public API - functions that external scripts should use
__all__ = [
    'get_device',
    'get_optimal_dtype',
    'optimize_pipeline',
    'create_generator',
    'load_loras',
    'load_model'
]

# Module-level logger
logger = logging_utils.get_logger(__name__)

# Type variable for available model types
AvailableModels = TypeVar('AvailableModels', StableDiffusionXLPipeline,
                          StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline)


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


def load_loras(pipeline: DiffusionPipeline, loras: list, token: str) -> None:
    """ Load LoRA models into the pipeline with support for local .safetensors files """
    if not loras:
        return

    logger.info("Loading %d LoRA(s)", len(loras))
    adapter_names = []

    # Load all LoRAs with unique adapter names (don't fuse yet)
    for i, lora_config in enumerate(loras):
        adapter_name = f"lora_{i}"
        adapter_names.append(adapter_name)

        # Check if this is a local .safetensors file (ends with .safetensors AND no repo specified)
        if lora_config.lora.endswith('.safetensors') and not lora_config.repo:
            logger.info("Loading local LoRA file: %s (weight: %.2f) as adapter '%s'",
                        lora_config.lora, lora_config.weight, adapter_name)
            pipeline.load_lora_weights(
                lora_config.lora,
                adapter_name=adapter_name,
                token=token
            )
        elif lora_config.repo:
            # Load LoRA file from within a repository
            logger.info("Loading LoRA: %s/%s (weight: %.2f) as adapter '%s'",
                        lora_config.repo, lora_config.lora, lora_config.weight, adapter_name)
            pipeline.load_lora_weights(
                lora_config.repo,
                weight_name=lora_config.lora,
                adapter_name=adapter_name,
                token=token
            )
        else:
            # Load LoRA from a dedicated repository
            logger.info("Loading LoRA: %s (weight: %.2f) as adapter '%s'",
                        lora_config.lora, lora_config.weight, adapter_name)
            pipeline.load_lora_weights(
                lora_config.lora,
                adapter_name=adapter_name,
                token=token
            )

    # Fuse all LoRAs simultaneously with their individual weights
    logger.info("Fusing all LoRAs simultaneously")
    for i, lora_config in enumerate(loras):
        pipeline.fuse_lora(
            adapter_names=[f"lora_{i}"],
            lora_scale=lora_config.weight
        )


def load_model(model_cls: Type[AvailableModels], model_id: str, dtype: torch.dtype, device: str, token: str) -> AvailableModels:
    """ Loads the given model class with automatic detection of single file vs directory models """

    # Check if this is a .safetensors file path
    if model_id.endswith('.safetensors'):
        logger.info("Detected .safetensors file, using from_single_file() method")
        try:
            return _load_model_from_single_file(model_cls, model_id, dtype, device)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to load model from single file: %s", e)
            raise e

    # For Hugging Face model IDs or directories, use from_pretrained
    logger.info("Using from_pretrained() method for model: %s", model_id)
    try:
        return _load_model_from_pretrained(model_cls, model_id, dtype, device, token)
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Check if this is a FP16 file naming issue
        error_str = str(e)
        if _is_fp16_file_error(error_str):

            logger.info("Detected FP16 model file naming issue, fixing links...")

            try:
                # Extract model cache path from error message
                model_path = _extract_model_path_from_error(error_str)

                # Fix FP16 model files automatically
                _fix_fp16_model_files(model_path)

                # Retry with same model_id - will use cached files
                return _load_model_from_pretrained(model_cls, model_id, dtype, device, token)
            except Exception as fix_error:
                logger.error("Failed to fix FP16 model files: %s", fix_error)
                raise e from fix_error
        else:
            # Re-raise if it's not a file naming issue
            raise e


# =============================================================================
# PRIVATE HELPER FUNCTIONS
# =============================================================================

def _fix_fp16_model_files(model_path: str) -> None:
    """ 
    Automatically creates symbolic links for FP16 model files to fix compatibility issues.
    Some SDXL models use .fp16.safetensors files but diffusers expects .safetensors files.
    """
    print(f"Fixing FP16 model files in: {model_path}")
    model_dir = Path(model_path)

    if not model_dir.exists():
        raise ValueError(f"Model directory does not exist: {model_path}")

    # Components that might have FP16 files
    components = ['text_encoder', 'text_encoder_2', 'vae', 'unet']
    links_created = 0

    for component in components:
        component_dir = model_dir / component
        logger.debug("Checking component: %s", component_dir)

        if not component_dir.exists():
            logger.debug("Component directory does not exist: %s", component_dir)
            continue

        # Check for FP16 files and create symbolic links
        fp16_files = list(component_dir.glob('*.fp16.safetensors'))
        logger.debug("Found %d FP16 files in %s: %s", len(fp16_files), component, [f.name for f in fp16_files])

        for fp16_file in fp16_files:
            # Create the expected filename without .fp16
            expected_file = component_dir / fp16_file.name.replace('.fp16.safetensors', '.safetensors')
            logger.debug("Checking if %s exists", expected_file)

            # Only create link if the expected file doesn't exist
            if not expected_file.exists():
                try:
                    os.symlink(fp16_file.name, expected_file)
                    logger.info("Created symbolic link: %s -> %s", expected_file, fp16_file.name)
                    links_created += 1
                except OSError as e:
                    logger.error("Failed to create symbolic link for %s: %s", fp16_file.name, e)
                    raise
            else:
                logger.debug("Symbolic link already exists: %s", expected_file)

    logger.info("FP16 fix completed. Created %d symbolic links.", links_created)


def _is_fp16_file_error(error_str: str) -> bool:
    """ Check if error is related to FP16 file naming issues """
    return ("no file named" in error_str and (".safetensors" in error_str or ".bin" in error_str)) or \
           ("diffusion_pytorch_model.safetensors" in error_str) or \
           ("model.safetensors" in error_str) or \
           ("diffusion_pytorch_model.bin" in error_str)


def _extract_model_path_from_error(error_message: str) -> str:
    """ Extract the model cache path from a Hugging Face error message """

    # Look for pattern like: "found in directory /path/to/models--repo--name/snapshots/hash/component"
    pattern = r"found in directory ([^\s]+)"
    match = re.search(pattern, error_message)

    if match:
        # Get the directory path and go up to the model root (snapshots/hash)
        component_path = match.group(1)
        model_path = str(Path(component_path).parent)  # Only go up one level to get snapshots/hash
        return model_path

    raise ValueError(f"Could not extract model path from error: {error_message}")


def _load_model_from_pretrained(model_cls: Type[AvailableModels], model_id: str, dtype: torch.dtype, device: str, token: str) -> AvailableModels:
    """ Load model using from_pretrained with standard parameters """
    return model_cls.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        token=token,
        add_watermarker=False
    ).to(device)


def _load_model_from_single_file(model_cls: Type[AvailableModels], model_path: str, dtype: torch.dtype, device: str) -> AvailableModels:
    """ Load model from a single .safetensors file using from_single_file """
    return model_cls.from_single_file(
        model_path,
        torch_dtype=dtype,
        use_safetensors=True,
        add_watermarker=False
    ).to(device)
