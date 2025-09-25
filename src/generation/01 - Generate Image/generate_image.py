#!/usr/bin/env python3
"""
Stable Diffusion XL 2-Step Image Generator
Generates a single image from a text prompt using the SDXL Base + Refiner pipeline
"""

import re
import os
import sys
from pathlib import Path
from typing import Type, TypeVar, Final

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, EulerDiscreteScheduler

# Clean imports using the utilities package
from pipeline_utilities import authentication, generation, args, logging_utils, paths
from pipeline_sdxl import sdxl_utils as sdxl

# To get good results we need to use a square aspect ratio so ensure that's the case


class _DefaultProperties:
    OUTPUT_WIDTH: Final[int] = 1024
    OUTPUT_HEIGHT: Final[int] = OUTPUT_WIDTH


_args = args.parse_arguments("Generates an image using Stable Diffusion XL")

_authentication = authentication.load_authentication_config(_args.authentication)
_config = generation.load_generation_config(_args.config)

_logger = logging_utils.setup_pipeline_logging(
    log_file=_args.log_file,
    debug=_config.data.debug
)

_AvailableModels = TypeVar('_AvailableModels', StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline)


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
        _logger.debug("Checking component: %s", component_dir)

        if not component_dir.exists():
            _logger.debug("Component directory does not exist: %s", component_dir)
            continue

        # Check for FP16 files and create symbolic links
        fp16_files = list(component_dir.glob('*.fp16.safetensors'))
        _logger.debug("Found %d FP16 files in %s: %s", len(fp16_files), component, [f.name for f in fp16_files])

        for fp16_file in fp16_files:
            # Create the expected filename without .fp16
            expected_file = component_dir / fp16_file.name.replace('.fp16.safetensors', '.safetensors')
            _logger.debug("Checking if %s exists", expected_file)

            # Only create link if the expected file doesn't exist
            if not expected_file.exists():
                try:
                    os.symlink(fp16_file.name, expected_file)
                    _logger.info("Created symbolic link: %s -> %s", expected_file, fp16_file.name)
                    links_created += 1
                except OSError as e:
                    _logger.error("Failed to create symbolic link for %s: %s", fp16_file.name, e)
                    raise
            else:
                _logger.debug("Symbolic link already exists: %s", expected_file)

    _logger.info("FP16 fix completed. Created %d symbolic links.", links_created)


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


def _load_model_from_pretrained(model_cls: Type[_AvailableModels], model_id: str, dtype: torch.dtype, device: str) -> _AvailableModels:
    """ Load model using from_pretrained with standard parameters """
    return model_cls.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        token=_authentication.data.huggingface,
        add_watermarker=False
    ).to(device)


def _load_image_model(model_cls: Type[_AvailableModels], model_id: str, dtype: torch.dtype, device: str) -> _AvailableModels:
    """ Loads the given model class with automatic FP16 file compatibility fixes """

    # Try to load the model normally first
    try:
        return _load_model_from_pretrained(model_cls, model_id, dtype, device)
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Check if this is a FP16 file naming issue
        error_str = str(e)
        if _is_fp16_file_error(error_str):

            _logger.info("Detected FP16 model file naming issue, fixing links...")

            try:
                # Extract model cache path from error message
                model_path = _extract_model_path_from_error(error_str)

                # Fix FP16 model files automatically
                _fix_fp16_model_files(model_path)

                # Retry with same model_id - will use cached files
                return _load_model_from_pretrained(model_cls, model_id, dtype, device)
            except Exception as fix_error:
                _logger.error("Failed to fix FP16 model files: %s", fix_error)
                raise e from fix_error
        else:
            # Re-raise if it's not a file naming issue
            raise e


def _generate_image() -> Path:
    """ Generate an image using SDXL Base + Refiner pipelines with a 2-step denoising process """

    # Ensure output folder exists
    output_path = (
        Path(_args.output) /
        paths.Paths.RESULT /
        paths.Paths.TEMP /
        paths.Paths.OUTPUT_01
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _logger.header("Setting up Stable Diffusion XL (Base + Refiner)")

    # Determine device and preferred precision
    device = sdxl.get_device()
    dtype = sdxl.get_optimal_dtype(device)

    # Get model IDs from configuration
    base_model_id = _config.data.generation.image.base_checkpoints
    refiner_model_id = _config.data.generation.image.refiner_checkpoints

    # Load Base Pipeline: generates coarse image latents
    _logger.info("Loading SDXL base pipeline (this will take a while the first time)")
    pipe = _load_image_model(StableDiffusionXLPipeline, base_model_id, dtype, device)

    # Load LoRAs for base pipeline
    sdxl.load_loras(pipe, _config.data.generation.image.base_loras, _authentication.data.huggingface)

    # Load Refiner Pipeline: refines the latents into a final high-quality image
    _logger.info("Loading SDXL refiner pipeline")
    refiner = _load_image_model(StableDiffusionXLImg2ImgPipeline, refiner_model_id, dtype, device)

    # Load LoRAs for refiner pipeline
    sdxl.load_loras(refiner, _config.data.generation.image.refiner_loras, _authentication.data.huggingface)

    # Optimize both pipelines for memory and compute efficiency
    sdxl.optimize_pipeline(pipe, device)
    sdxl.optimize_pipeline(refiner, device)

    # Ensure consistent scheduler (Euler) for both pipelines
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    refiner.scheduler = EulerDiscreteScheduler.from_config(refiner.scheduler.config)

    _logger.header("Generating Image (2-Step)")

    with torch.no_grad():

        generator = sdxl.create_generator(device, _config.data.generation.seed)

        _logger.debug("Using generation properties:")
        _logger.debug("  Device: %s", device)
        _logger.debug("  Seed: %d", _config.data.generation.seed)
        _logger.debug("  Generation Steps: %d", _config.data.generation.image.steps)
        _logger.debug("  Base Fractal: %.2f", _config.data.generation.image.base_fractal)
        _logger.debug("  Guidance Scale: %.1f", _config.data.generation.image.guidance)
        _logger.debug("  Base Checkpoint: %s", base_model_id)
        _logger.debug("  Refiner Checkpoint: %s", refiner_model_id)

        # Pre-encode prompts to support longer prompts beyond 77 token limit
        _logger.info("Encoding prompts for generation")
        positive_embeds = sdxl.encode_long_prompt(pipe, _config.data.prompts.image_positive)
        negative_embeds = sdxl.encode_long_prompt(pipe, _config.data.prompts.image_negative)

        # Step 1: Base model generates coarse latents
        _logger.info("Running base model for %.1f%% of steps", _config.data.generation.image.base_fractal * 100)
        latents = pipe(
            prompt_embeds=positive_embeds,
            negative_prompt_embeds=negative_embeds,
            width=_DefaultProperties.OUTPUT_WIDTH,
            height=_DefaultProperties.OUTPUT_HEIGHT,
            num_inference_steps=_config.data.generation.image.steps,
            denoising_end=_config.data.generation.image.base_fractal,
            guidance_scale=_config.data.generation.image.guidance,
            generator=generator,
            output_type="latent"  # Output latents, not a PIL image
        ).images

        # Step 2: Refiner model completes image generation
        _logger.info("Running refiner model for the remaining %.1f%% of steps",
                     (1 - _config.data.generation.image.base_fractal) * 100)
        result = refiner(
            prompt_embeds=positive_embeds,
            negative_prompt_embeds=negative_embeds,
            image=latents,  # Use base model latents as input
            num_inference_steps=_config.data.generation.image.steps,
            denoising_start=_config.data.generation.image.base_fractal,
            guidance_scale=_config.data.generation.image.guidance,
            generator=generator,
            output_type="pil"  # Final output is PIL image
        )
        image = result.images[0]

    # Save final image
    image.save(output_path)

    return output_path


def _main() -> None:
    """ Main entry point """
    try:
        output_path = _generate_image()
        _logger.info("Success! Image generated: %s", output_path)
    except ImportError as e:
        _logger.error("Failed to import required libraries: %s", e)
        sys.exit(1)
    except OSError as e:
        _logger.error("File system error (check permissions and disk space): %s", e)
        sys.exit(1)
    except RuntimeError as e:
        _logger.error("Runtime error during image generation: %s", e)
        sys.exit(1)
    except ValueError as e:
        _logger.error("Configuration or input validation error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    _main()
