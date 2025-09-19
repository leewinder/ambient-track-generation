#!/usr/bin/env python3
"""
Stable Diffusion Out Painter
Takes a pre-generated image and widens it to support 16:9
"""
import sys
from pathlib import Path
from PIL import Image
import torch
from diffusers import StableDiffusionXLInpaintPipeline

# Clean imports using the utilities package
from pipeline_utilities import authentication, generation, args, logging_utils
from pipeline_utilities import sdxl_utils as sdxl


class _Side:
    """ Constants identifying each side of the image """
    LEFT = "left"
    RIGHT = "right"


class _Dimensions:
    """ Container for image dimensions used in outpainting process """

    def __init__(self, source_width: int, source_height: int,
                 target_width: int, target_height: int,
                 working_width: int, working_height: int) -> None:
        self.source = _DimensionPair(source_width, source_height)
        self.target = _DimensionPair(target_width, target_height)
        self.working = _DimensionPair(working_width, working_height)


class _DimensionPair:
    """ Container for width and height pair """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height


_args = args.parse_arguments("Outpaints a given image using Stable Diffusion")

_authentication = authentication.load_authentication_config(_args.authentication)
_config = generation.load_generation_config(_args.config)

# Setup logging using config debug flag
_logger = logging_utils.setup_pipeline_logging(
    log_file=_args.log_file,
    debug=_config.data.debug
)


def _calculate_target_dimensions(source_width: int, source_height: int) -> _Dimensions:
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

    return _Dimensions(
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height,
        working_width=working_width,
        working_height=working_height
    )


def _prime_canvas_with_smear(source_image: Image.Image, dimensions: _Dimensions) -> Image.Image:
    """ Creates a new canvas and primes it for outpainting by smearing the edges of the source image """
    working_expansion_per_side = (dimensions.working.width - dimensions.source.width) // 2

    _logger.info("Creating and priming the canvas for outpainting")

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

    _save_interim_result(final_canvas, "widened")

    return final_canvas

def _outpaint_side(side: _Side, dimensions: _Dimensions, generator: torch.Generator, pipeline: StableDiffusionXLInpaintPipeline, canvas: Image.Image) -> Image.Image:
    """ Generates an out painted side for the image """

    # Build up the properties of the mask side
    working_expansion_per_side = (dimensions.working.width - dimensions.source.width) // 2
    feather_size = max(1, int(dimensions.source.width * (_config.data.generation.outpaint.feathering / 100.0)))
    mask_width = working_expansion_per_side + feather_size

    # Create the default mas
    mask = Image.new("L", (dimensions.working.width, dimensions.working.height), 0)  # Start with black (keep all)
    mask_area = Image.new("L", (mask_width, dimensions.working.height), 255)  # Create a solid white area to paint in

    # We do have a bit of per side logic, but most of it is shared
    mask_offset = 0
    if side == _Side.LEFT:
        for x_offset in range(feather_size):
            # Calculate x from the right edge of the area
            x = mask_width - feather_size + x_offset
            alpha = int(255 * (1.0 - (x_offset / feather_size)))
            for y in range(dimensions.working.height):
                mask_area.putpixel((x, y), alpha)
    else:
        mask_offset = (dimensions.source.width + working_expansion_per_side) - feather_size
        for x in range(feather_size):
            alpha = int(255 * (x / feather_size))
            for y in range(dimensions.working.height):
                mask_area.putpixel((x, y), alpha)

    # Paste the feathered white area onto the original canvas
    mask.paste(mask_area, (mask_offset, 0))
    _save_interim_result(mask, f"{side}_mask")


    # Now run the out painting step across the unmasked area
    result: Image.Image = None
    with torch.no_grad():
        result = pipeline(
            prompt=_config.data.prompts.image_positive,
            negative_prompt=_config.data.prompts.image_negative,
            image=canvas,
            mask_image=mask,
            num_inference_steps=_config.data.generation.outpaint.steps,
            guidance_scale=_config.data.generation.outpaint.guidance,
            generator=generator,
            height=dimensions.working.height,
            width=dimensions.working.width
        ).images[0]
    _save_interim_result(result, f"{side}_result")
    return result


def _save_interim_result(interim_image: Image.Image, name: str) -> None:
    """ Outputs the interim images generated by the process if debugging is enabled """

    if _config.data.debug:
        internal_temp_folder = "02 - Widen Image"
        interim_path = (
            Path(_args.output) /
            _config.data.paths.result_dir /
            _config.data.paths.temp_dir /
            internal_temp_folder /
            f"{name}.png"
        )
        interim_path.parent.mkdir(parents=True, exist_ok=True)

        interim_image.save(interim_path)
        _logger.debug("Saved interim result: %s", interim_path)


def _outpaint_image() -> str:
    """ Generates a new out painted image from a pre-created image """

    # Verify our source image exists
    source_image_path = (
        Path(_args.output) /
        _config.data.paths.result_dir /
        _config.data.paths.temp_dir /
        _config.data.paths.outputs.stage_01
    )
    if not source_image_path.is_file():
        raise FileNotFoundError(f"Required source image not found: {source_image_path}")

    _logger.header("Setting up Stable Diffusion Inpainting")
    _logger.info("Loading SDXL inpainting pipeline (this will take a while the first time)")
    
    # Set up our inpainting pipeline
    device = sdxl.get_device()
    dtype = sdxl.get_optimal_dtype(device)
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=dtype,
        use_safetensors=True,
        token=_authentication.data.huggingface,
        add_watermarker=False
    ).to(device)
    sdxl.optimize_pipeline(pipe, device)

    # Calculate target and working dimensions for 16:9 ratio from the source image
    _logger.header("Creating Canvas")
    source_image = Image.open(source_image_path).convert("RGB")
    source_width, source_height = source_image.size
    dimensions = _calculate_target_dimensions(source_width, source_height)
    working_canvas = _prime_canvas_with_smear(source_image, dimensions)

    _logger.debug("The following properties will be used as part of the outpainting process")
    _logger.debug("  Source image: %dx%d", dimensions.source.width, dimensions.source.height)
    _logger.debug("  Target image: %dx%d", dimensions.target.width, dimensions.target.height)
    _logger.debug("  Working image: %dx%d", dimensions.working.width, dimensions.working.height)
    _logger.debug("  Feather size: %.1f%%", _config.data.generation.outpaint.feathering)
    _logger.debug("  Steps: %d", _config.data.generation.outpaint.steps)
    _logger.debug("  Guidance: %.1f", _config.data.generation.outpaint.guidance)

    generator = sdxl.create_generator(device, _config.data.generation.seed)
    _logger.info("Outpainting right side")
    working_canvas = _outpaint_side(_Side.RIGHT, dimensions, generator, pipe, working_canvas)
    
    _logger.info("Outpainting left side")
    working_canvas = _outpaint_side(_Side.LEFT, dimensions, generator, pipe, working_canvas)

    # Save the final result
    final_output_path = (
        Path(_args.output) / 
        _config.data.paths.result_dir /
        _config.data.paths.temp_dir / 
        _config.data.paths.outputs.stage_02
    )
    working_canvas.save(final_output_path, quality=_config.data.generation.outpaint.save_quality)

    return str(final_output_path)


def _main() -> None:
    """ Main entry point """
    try:
        output_path = _outpaint_image()
        _logger.info("Success! Image out painted: %s", output_path)
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
