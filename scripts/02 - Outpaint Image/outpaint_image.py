#!/usr/bin/env python3
"""
Stable Diffusion Out Painter
Takes a pre-generated image and out paints it to 16:9
"""
import sys
import logging
from pathlib import Path
from PIL import Image
import torch
from diffusers import StableDiffusionXLInpaintPipeline


class Dimensions:
    """ Container for image dimensions used in outpainting process """

    def __init__(self, source_width: int, source_height: int,
                 target_width: int, target_height: int,
                 working_width: int, working_height: int) -> None:
        self.source = DimensionPair(source_width, source_height)
        self.target = DimensionPair(target_width, target_height)
        self.working = DimensionPair(working_width, working_height)


class DimensionPair:
    """ Container for width and height pair """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height


# Add path and import project-specific config and authentication utilities
common_path = Path(__file__).parent.parent / "00 - Common"
sys.path.insert(0, str(common_path))

# fmt: off
# noqa: E402,E501  # pylint: disable-next=import-error,wrong-import-position,multiple-imports
import authentication, config, args
# fmt: on

authentication = authentication.load_authentication()
config = config.load_config()
args = args.parse_common_arguments("Outpaints a given image using Stable Diffusion")

# Suppress the informational warnings from the diffusers library
logging.getLogger('diffusers').setLevel(logging.ERROR)


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


def calculate_target_dimensions(source_width: int, source_height: int) -> Dimensions:
    """ Calculate target and working dimensions for 16:9 aspect ratio outpainting """
    # For 16:9 aspect ratio: width = height * 16 / 9
    target_width = int(source_height * 16 / 9)  # Keep exact 16:9 ratio (1820)
    target_height = source_height  # Keep height the same

    # Calculate working dimensions (5% extra on each side for better generation)
    extra_width = int(target_width * 0.05)  # 5% of target width
    working_width = target_width + (2 * extra_width)  # 5% on left + 5% on right
    working_height = target_height  # Height stays the same

    # Ensure working width is multiple of 8 (for pipeline compatibility)
    # Round UP to next multiple of 8 to ensure we have enough space
    working_width = ((working_width + 7) // 8) * 8

    return Dimensions(
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height,
        working_width=working_width,
        working_height=working_height
    )


def create_feathered_mask(width: int, height: int, mask_width: int,
                          feather_size: int, mask_x: int = 0,
                          feather_from_left: bool = True) -> Image.Image:
    """ Create a feathered mask for outpainting """
    mask = Image.new("L", (width, height), 0)  # Start with black (keep all)
    mask_area = Image.new("L", (mask_width, height), 255)  # Create a solid white area to inpaint

    if feather_size > 0:
        if feather_from_left:
            # Right-side mask: Feather the LEFT edge of mask_area (fades 0 -> 255)
            for x in range(feather_size):
                alpha = int(255 * (x / feather_size))
                for y in range(height):
                    mask_area.putpixel((x, y), alpha)
        else:
            # Left-side mask: Feather the RIGHT edge of mask_area (fades 255 -> 0)
            for x_offset in range(feather_size):
                # Calculate x from the right edge of the area
                x = mask_width - feather_size + x_offset
                alpha = int(255 * (1.0 - (x_offset / feather_size)))
                for y in range(height):
                    mask_area.putpixel((x, y), alpha)

    # Paste the feathered white area onto the black canvas
    mask.paste(mask_area, (mask_x, 0))

    return mask


def prime_canvas_with_smear(source_image: Image.Image, dimensions: Dimensions) -> Image.Image:
    """ Creates a new canvas and primes it for outpainting by smearing the edges of the source image """
    working_expansion_per_side = (dimensions.working.width - dimensions.source.width) // 2

    print("Creating and priming the canvas for outpainting...")

    # Create a new canvas and paste the source image into the center
    final_canvas = Image.new("RGB", (dimensions.working.width, dimensions.working.height))
    final_canvas.paste(source_image, (working_expansion_per_side, 0))

    # --- Prime the canvas by smearing the edges ---
    # Smear the right edge
    right_edge_column = source_image.crop(
        (dimensions.source.width - 1, 0, dimensions.source.width, dimensions.working.height))
    smeared_right = right_edge_column.resize(
        (working_expansion_per_side, dimensions.working.height), Image.Resampling.BOX)
    final_canvas.paste(smeared_right, (dimensions.source.width + working_expansion_per_side, 0))

    # Smear the left edge
    left_edge_column = source_image.crop((0, 0, 1, dimensions.working.height))
    smeared_left = left_edge_column.resize(
        (working_expansion_per_side, dimensions.working.height), Image.Resampling.BOX)
    final_canvas.paste(smeared_left, (0, 0))

    return final_canvas


def save_interim_result(interim_image: Image.Image, name: str) -> None:
    """ Outputs the interim images generated by the process if debugging is enabled """

    if config.debug:
        internal_temp_folder = "outpaint"
        interim_path = Path(args.output) / config.result_dir / config.temp_dir / internal_temp_folder / f"{name}.png"
        interim_path.parent.mkdir(parents=True, exist_ok=True)
        interim_image.save(interim_path)


def outpaint_image() -> str:
    """ Generates a new out painted image from a pre-created image """

    print("\n\n-------- IDENTIFYING SOURCE IMAGE AND OUTPUT --------")

    # Verify our source image exists
    source_image_path = Path(args.output) / config.result_dir / config.temp_dir / config.output_stage_01
    if not source_image_path.is_file():
        raise FileNotFoundError(f"Required input file not found: {source_image_path}")

    # Load the source image
    source_image = Image.open(source_image_path).convert("RGB")
    source_width, source_height = source_image.size

    # Calculate target and working dimensions for 16:9 ratio
    dimensions = calculate_target_dimensions(source_width, source_height)

    # Calculate how much to expand on each side for working dimensions
    working_expansion_per_side = (dimensions.working.width - dimensions.source.width) // 2

    # Calculate feather size (2% of image width)
    feather_size = max(1, int(dimensions.source.width * (config.outpaint_feathering / 100.0)))

    print("The following properties will be used as part of the outpainting process")
    print(f"  * Source image: {dimensions.source.width}x{dimensions.source.height}")
    print(f"  * Target image: {dimensions.target.width}x{dimensions.target.height}")
    print(f"  * Working image: {dimensions.working.width}x{dimensions.working.height}")
    print(f"  * Working expansion per side: {working_expansion_per_side}px")
    print(f"  * Feather size: {config.outpaint_feathering}%")
    print(f"  * Steps: {config.outpaint_generation_steps}")
    print(f"  * Guidance: {config.outpaint_guidance}")

    print("\n\n-------- SETTING UP STABLE DIFFUSION INPAINTING PIPELINE --------")
    print("Loading SDXL inpainting pipeline (this will take a while the first time)...")
    # Determine device and preferred precision
    device = get_device()
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device == "cuda" else torch.float32
    )

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=dtype,
        use_safetensors=True,
        token=authentication.huggingface_token,
        add_watermarker=False
    ).to(device)

    # Optimize pipeline for memory and compute efficiency
    if device == "cuda":
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    print("\n\n-------- OUT PAINTING IMAGE --------")

    # Create and prime the canvas using our new function
    final_canvas = prime_canvas_with_smear(source_image, dimensions)
    save_interim_result(final_canvas, "widened")

    print("Outpainting right side...")

    # Create mask for right side (mask should cover the right expansion area + overlap)
    # Extend mask into original image area for proper blending
    right_mask_width = working_expansion_per_side + feather_size
    right_mask_x = (dimensions.source.width + working_expansion_per_side) - feather_size
    right_mask = create_feathered_mask(dimensions.working.width, dimensions.working.height,
                                       right_mask_width, feather_size,
                                       mask_x=right_mask_x,
                                       feather_from_left=True)
    save_interim_result(right_mask, "right_mask")

    # Perform right side inpainting
    with torch.no_grad():
        generator = torch.Generator(device=device).manual_seed(config.generation_seed)

        right_result = pipe(
            prompt=config.image_prompt_positive,
            negative_prompt=config.image_prompt_negative,
            image=final_canvas,
            mask_image=right_mask,
            num_inference_steps=config.outpaint_generation_steps,
            guidance_scale=config.outpaint_guidance,
            generator=generator,
            height=dimensions.working.height,
            width=dimensions.working.width
        ).images[0]
    save_interim_result(right_result, "right_result")

    print("Outpainting left side...")

    # Create mask for left side (mask should cover the left expansion area + overlap)
    # Extend mask into original image area for proper blending
    left_mask_width = working_expansion_per_side + feather_size
    left_mask = create_feathered_mask(dimensions.working.width, dimensions.working.height,
                                      left_mask_width, feather_size,
                                      mask_x=0, feather_from_left=False)
    save_interim_result(left_mask, "left_mask")

    # Perform left side inpainting
    with torch.no_grad():
        left_result = pipe(
            prompt=config.image_prompt_positive,
            negative_prompt=config.image_prompt_negative,
            image=right_result,
            mask_image=left_mask,
            num_inference_steps=config.outpaint_generation_steps,
            guidance_scale=config.outpaint_guidance,
            generator=generator,
            height=dimensions.working.height,
            width=dimensions.working.width
        ).images[0]
    save_interim_result(left_result, "left_result")

    # Crop the result to exact 16:9 ratio (remove the 5% extra on each side)
    print("Cropping to exact 16:9 ratio...")
    crop_x = (dimensions.working.width - dimensions.target.width) // 2
    crop_box = (crop_x, 0, crop_x + dimensions.target.width, dimensions.target.height)
    final_result = left_result.crop(crop_box)

    # Save the final result
    final_output_path = Path(args.output) / config.result_dir / config.temp_dir / config.output_stage_02
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    final_result.save(final_output_path)

    return str(final_output_path)


def main() -> None:
    """ Main entry point """
    try:
        output_path = outpaint_image()
        print(f"\nSuccess! Image out painted: {output_path}")
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
