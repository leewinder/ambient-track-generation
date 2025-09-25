#!/usr/bin/env python3
"""
Stable Diffusion XL 2-Step Image Generator
Generates a single image from a text prompt using the SDXL Base + Refiner pipeline
"""

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


def _load_image_model(model_cls: Type[_AvailableModels], model_id: str, dtype: torch.dtype, device: str) -> _AvailableModels:
    """ Loads the given model class """
    return model_cls.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        token=_authentication.data.huggingface,
        add_watermarker=False
    ).to(device)


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

    # Define the model IDs for base and refiner pipelines
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"

    # Load Base Pipeline: generates coarse image latents
    _logger.info("Loading SDXL base pipeline (this will take a while the first time)")
    pipe = _load_image_model(StableDiffusionXLPipeline, base_model_id, dtype, device)

    # Load Refiner Pipeline: refines the latents into a final high-quality image
    _logger.info("Loading SDXL refiner pipeline")
    refiner = _load_image_model(StableDiffusionXLImg2ImgPipeline, refiner_model_id, dtype, device)

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

        # Step 1: Base model generates coarse latents
        _logger.info("Running base model for %.1f%% of steps", _config.data.generation.image.base_fractal * 100)
        latents = pipe(
            prompt=_config.data.prompts.image_positive,
            negative_prompt=_config.data.prompts.image_negative,
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
            prompt=_config.data.prompts.image_positive,
            negative_prompt=_config.data.prompts.image_negative,
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
