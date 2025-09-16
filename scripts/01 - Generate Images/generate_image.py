#!/usr/bin/env python3
"""
Stable Diffusion XL 2-Step Image Generator
Generates a single image from a text prompt using the SDXL Base + Refiner pipeline.
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
import authentication, config, args
# fmt: on

authentication = authentication.load_authentication()
config = config.load_config()
args = args.parse_common_arguments("Generates an image using Stable Diffusion XL")


def get_device():
    """Select the best available compute device: MPS (Apple), CUDA (NVIDIA), or CPU fallback."""
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) for acceleration")
        return "mps"
    if torch.cuda.is_available():
        print("Using CUDA for acceleration")
        return "cuda"
    print("Using CPU (no GPU acceleration available)")
    return "cpu"


def generate_image():
    """Generate an image using SDXL Base + Refiner pipelines with a 2-step denoising process."""

    # Ensure output folder exists
    output_path = Path(args.output) / config.result_dir / config.temp_dir / config.output_stage_01
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n\n-------- SETTING UP STABLE DIFFUSION XL (Base + Refiner) --------")

    # Determine device and preferred precision
    device = get_device()
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device == "cuda" else torch.float32
    )

    # Define the model IDs for base and refiner pipelines
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"

    print("Loading SDXL base pipeline (this will take a while the first time)...")
    # Load Base Pipeline: generates coarse image latents
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        token=authentication.huggingface_token,
        add_watermarker=False
    ).to(device)

    print("Loading SDXL refiner pipeline...")
    # Load Refiner Pipeline: refines the latents into a final high-quality image
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        refiner_model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        token=authentication.huggingface_token,
        add_watermarker=False
    ).to(device)

    # Optimize both pipelines for memory and compute efficiency
    if device == "cuda":
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        if hasattr(refiner, "enable_model_cpu_offload"):
            refiner.enable_model_cpu_offload()

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(refiner, "enable_attention_slicing"):
        refiner.enable_attention_slicing()

    # Ensure consistent scheduler (Euler) for both pipelines
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    refiner.scheduler = EulerDiscreteScheduler.from_config(refiner.scheduler.config)

    print("\n\n-------- GENERATING IMAGE (2-Step) --------")

    with torch.no_grad():

        generator = torch.Generator(device=device).manual_seed(config.generation_seed)

        print("Using generation properties of:")
        print(f"  * Seed: {config.generation_seed}")
        print(f"  * Generation Steps: {config.image_generation_steps}")
        print(f"  * Base Fractal: {config.image_base_fractal}")
        print(f"  * Guidance Scale: {config.image_guidance}")

        # Step 1: Base model generates coarse latents
        print(f"Running base model for {config.image_base_fractal * 100}% of steps...")
        latents = pipe(
            prompt=config.image_prompt_positive,
            negative_prompt=config.image_prompt_negative,
            width=config.image_width,
            height=config.image_height,
            num_inference_steps=config.image_generation_steps,
            denoising_end=config.image_base_fractal,
            guidance_scale=config.image_guidance,
            generator=generator,
            output_type="latent"  # Output latents, not a PIL image
        ).images

        # Step 2: Refiner model completes image generation
        print(f"Running refiner model for the remaining {(1 - config.image_base_fractal) * 100}% of steps...")
        result = refiner(
            prompt=config.image_prompt_positive,
            negative_prompt=config.image_prompt_negative,
            image=latents,  # Use base model latents as input
            num_inference_steps=config.image_generation_steps,
            denoising_start=config.image_base_fractal,
            guidance_scale=config.image_guidance,
            generator=generator,
            output_type="pil"  # Final output is PIL image
        )
        image = result.images[0]

    # Save final image
    image.save(output_path, quality=95)
    print(f"\nImage saved to: {output_path}")

    return output_path


def main():
    """Main entry point."""
    try:
        output_path = generate_image()
        print(f"\nSuccess! Image generated: {output_path}")
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
