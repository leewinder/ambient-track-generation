#!/usr/bin/env python3
"""
Stable Diffusion XL 2-Step Image Generator
Generates a single image from a text prompt using the SDXL Base + Refiner pipeline
"""

import sys
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, EulerDiscreteScheduler


# Add path and import project-specific config and authentication utilities
common_path = Path(__file__).parent.parent / "00 - Common"
sys.path.insert(0, str(common_path))

# fmt: off
# noqa: E402,E501  # pylint: disable-next=import-error,wrong-import-position,multiple-imports
import authentication, config, args, sdxl_utils as sdxl, logging_utils
# fmt: on

authentication = authentication.load_authentication()
config = config.load_config()
args = args.parse_arguments("Generates an image using Stable Diffusion XL")

# Setup logging using config debug flag
logger = logging_utils.setup_pipeline_logging(
    log_file=args.log_file,
    debug=config.data.debug,
    script_name="Image Generation"
)


def generate_image() -> Path:
    """ Generate an image using SDXL Base + Refiner pipelines with a 2-step denoising process """

    # Ensure output folder exists
    output_path = Path(args.output) / config.data.paths.result_dir / \
        config.data.paths.temp_dir / config.data.paths.outputs.stage_01
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.header("Setting up Stable Diffusion XL (Base + Refiner)")

    # Determine device and preferred precision
    device = sdxl.get_device()
    dtype = sdxl.get_optimal_dtype(device)

    # Define the model IDs for base and refiner pipelines
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"

    logger.info("Loading SDXL base pipeline (this will take a while the first time)")
    # Load Base Pipeline: generates coarse image latents
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        token=authentication.data.huggingface,
        add_watermarker=False
    ).to(device)

    logger.info("Loading SDXL refiner pipeline")
    # Load Refiner Pipeline: refines the latents into a final high-quality image
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        refiner_model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        token=authentication.data.huggingface,
        add_watermarker=False
    ).to(device)

    # Optimize both pipelines for memory and compute efficiency
    sdxl.optimize_pipeline(pipe, device)
    sdxl.optimize_pipeline(refiner, device)

    # Ensure consistent scheduler (Euler) for both pipelines
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    refiner.scheduler = EulerDiscreteScheduler.from_config(refiner.scheduler.config)

    logger.header("Generating Image (2-Step)")

    with torch.no_grad():

        generator = sdxl.create_generator(device, config.data.generation.seed)

        logger.debug("Using generation properties:")
        logger.debug("  Device: %s", device)
        logger.debug("  Seed: %d", config.data.generation.seed)
        logger.debug("  Generation Steps: %d", config.data.generation.image.steps)
        logger.debug("  Base Fractal: %.2f", config.data.generation.image.base_fractal)
        logger.debug("  Guidance Scale: %.1f", config.data.generation.image.guidance)

        # Step 1: Base model generates coarse latents
        logger.info("Running base model for %.1f%% of steps", config.data.generation.image.base_fractal * 100)
        latents = pipe(
            prompt=config.data.prompts.image_positive,
            negative_prompt=config.data.prompts.image_negative,
            width=config.data.dimensions.image.width,
            height=config.data.dimensions.image.height,
            num_inference_steps=config.data.generation.image.steps,
            denoising_end=config.data.generation.image.base_fractal,
            guidance_scale=config.data.generation.image.guidance,
            generator=generator,
            output_type="latent"  # Output latents, not a PIL image
        ).images

        # Step 2: Refiner model completes image generation
        logger.info("Running refiner model for the remaining %.1f%% of steps",
                    (1 - config.data.generation.image.base_fractal) * 100)
        result = refiner(
            prompt=config.data.prompts.image_positive,
            negative_prompt=config.data.prompts.image_negative,
            image=latents,  # Use base model latents as input
            num_inference_steps=config.data.generation.image.steps,
            denoising_start=config.data.generation.image.base_fractal,
            guidance_scale=config.data.generation.image.guidance,
            generator=generator,
            output_type="pil"  # Final output is PIL image
        )
        image = result.images[0]

    # Save final image
    image.save(output_path, quality=95)

    return output_path


def main() -> None:
    """ Main entry point """
    try:
        output_path = generate_image()
        logger.info("Success! Image generated: %s", output_path)
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
