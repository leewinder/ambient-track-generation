#!/usr/bin/env python3
"""
Up Scaler
Takes a pre-generated 1080p image and up scales it to 4k using Real-ESRGAN
"""

import gc
import sys
from typing import Final
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Import our local utilities
from pipeline_utilities import authentication, generation, args, logging_utils, paths


class _DefaultProperties:
    UPSCALE_FACTOR: Final[int] = 2
    MODEL_NAME: Final[str] = "RealESRGAN_x2plus"
    EXPECTED_INPUT_WIDTH: Final[int] = 1920
    EXPECTED_INPUT_HEIGHT: Final[int] = 1080
    EXPECTED_OUTPUT_WIDTH: Final[int] = 3840
    EXPECTED_OUTPUT_HEIGHT: Final[int] = 2160


_args = args.parse_arguments("Up scales a given image to 4k")
_authentication = authentication.load_authentication_config(_args.authentication)
_config = generation.load_generation_config(_args.config)
_logger = logging_utils.setup_pipeline_logging(
    log_file=_args.log_file,
    debug=_config.data.debug
)


def _get_device() -> torch.device:
    """ Get the best available device for processing """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _clear_memory() -> None:
    """ Clears system and MPS memory """
    gc.collect()
    if torch.backends.mps.is_available():
        # MPS cache clearing is only available in newer PyTorch versions
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()


def _validate_input_resolution(image: Image.Image) -> None:
    """ Validates that the input image is 1080p (1920x1080) """

    if image.width != _DefaultProperties.EXPECTED_INPUT_WIDTH or image.height != _DefaultProperties.EXPECTED_INPUT_HEIGHT:
        raise ValueError(
            f"Input image resolution is {image.width}x{image.height}, "
            f"but expected {_DefaultProperties.EXPECTED_INPUT_WIDTH}x{_DefaultProperties.EXPECTED_INPUT_HEIGHT} (1080p)"
        )
    _logger.info("Input resolution validation passed: %dx%d (1080p)", image.width, image.height)


def _validate_output_resolution(image: Image.Image) -> None:
    """ Validates that the output image is 4K (3840x2160) """

    if image.width != _DefaultProperties.EXPECTED_OUTPUT_WIDTH or image.height != _DefaultProperties.EXPECTED_OUTPUT_HEIGHT:
        raise ValueError(
            f"Output image resolution is {image.width}x{image.height}, "
            f"but expected {_DefaultProperties.EXPECTED_OUTPUT_WIDTH}x{_DefaultProperties.EXPECTED_OUTPUT_HEIGHT} (4K)"
        )
    _logger.info("Output resolution validation passed: %dx%d (4K)", image.width, image.height)


def _upscale_image() -> str:
    """ Up scales a given image to 4k using Real-ESRGAN """

    # Define input and output paths
    source_image_path = (
        Path(_args.output) /
        paths.Paths.RESULT /
        paths.Paths.TEMP /
        paths.Paths.OUTPUT_02
    )
    # Validate input image exists
    if not source_image_path.is_file():
        raise FileNotFoundError(f"Required source image not found: {source_image_path}")

    _logger.header("Setting up Real-ESRGAN")

    # Get device and setup Real-ESRGAN
    device = _get_device()

    # Create the Real-ESRGAN model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)

    # Initialize the upscaler
    upsampler = RealESRGANer(
        scale=_DefaultProperties.UPSCALE_FACTOR,
        model_path='../../../models/RealESRGAN_x2plus.pth',  # Download model automatically
        model=model,
        tile=0,  # No tiling for better quality
        tile_pad=10,
        pre_pad=0,
        half=False,  # Keep full precision for better quality
        gpu_id=None if str(device) == 'mps' else 0,  # Handle MPS device
        device=device
    )

    _logger.info("Loading source image: %s", source_image_path)
    source_image = Image.open(source_image_path).convert("RGB")

    # Validate input resolution is 1080p
    _validate_input_resolution(source_image)

    _logger.info("Starting upscaling process (2x upscale)")

    # Convert PIL image to numpy array for Real-ESRGAN
    img_array = np.array(source_image)

    # Perform upscaling
    with torch.no_grad():
        upscaled_array, _ = upsampler.enhance(img_array, outscale=_DefaultProperties.UPSCALE_FACTOR)

    # Convert back to PIL Image
    upscaled_image = Image.fromarray(upscaled_array)

    # Validate output resolution is 4K
    _validate_output_resolution(upscaled_image)

    # Save the upscaled image
    final_output_path = (
        Path(_args.output) /
        paths.Paths.RESULT /
        paths.Paths.TEMP /
        paths.Paths.OUTPUT_03
    )
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    upscaled_image.save(final_output_path)

    # Clean up memory
    del upsampler
    del model
    _clear_memory()

    _logger.info("Upscaling completed successfully")
    return str(final_output_path)


def main() -> None:
    """ Main entry point """
    try:
        output_path = _upscale_image()
        _logger.info("Success! Image up scaled: %s", output_path)
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        _logger.error("%s: %s", type(e).__name__, e)
        sys.exit(1)


if __name__ == "__main__":
    main()
