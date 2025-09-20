#!/usr/bin/env python3
"""
Stable Diffusion Out Painter
Takes a pre-generated image and expands it to 1080p
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from PIL import Image
import torch
from diffusers import StableDiffusionXLInpaintPipeline

# Clean imports using the utilities package
from pipeline_utilities import authentication, generation, args, logging_utils
from pipeline_utilities import sdxl_utils as sdxl


DEFAULT_IMAGE_WIDTH = 1920
DEFAULT_IMAGE_HEIGHT = 1080
CANVAS_EXPANSION_PERCENT = 10


class _Side:
    """ Constants identifying each side of the image """
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


@dataclass(frozen=True, slots=True)
class _DimensionPair:
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Dimensions must be positive")


@dataclass(frozen=True, slots=True)
class _Dimensions:
    source: _DimensionPair
    target: _DimensionPair
    working: _DimensionPair

    def __post_init__(self) -> None:
        if self.working.width < self.target.width or self.working.height < self.target.height:
            raise ValueError("Working dimensions must be larger than target dimensions")


_args = args.parse_arguments("Expands a given image using Stable Diffusion's inpainting model")

_authentication = authentication.load_authentication_config(_args.authentication)
_config = generation.load_generation_config(_args.config)

# Setup logging using config debug flag
_logger = logging_utils.setup_pipeline_logging(
    log_file=_args.log_file,
    debug=_config.data.debug
)


def _calculate_target_dimensions(source_width: int, source_height: int) -> _Dimensions:
    """ Calculate target and working dimensions with oversized working canvas for better outpainting """

    # Target dimensions are fixed 1080p
    target_width = DEFAULT_IMAGE_WIDTH
    target_height = DEFAULT_IMAGE_HEIGHT

    # Calculate working dimensions (CANVAS_EXPANSION_PERCENT extra on each side + up to the next multiple of 8)
    # This allows us to run at a multiple of 8, while also cropping down and removing the edge artifacts
    expansion_width = int(target_width * (CANVAS_EXPANSION_PERCENT / 100))
    expansion_height = int(target_height * (CANVAS_EXPANSION_PERCENT / 100))

    # Add padding to both dimensions
    working_width = target_width + (2 * expansion_width)
    working_height = target_height + (2 * expansion_height)

    # Round UP to next multiple of 8 to ensure compatibility
    working_width = ((working_width + 7) // 8) * 8
    working_height = ((working_height + 7) // 8) * 8

    return _Dimensions(
        source=_DimensionPair(source_width, source_height),
        target=_DimensionPair(target_width, target_height),
        working=_DimensionPair(working_width, working_height)
    )


def _prime_canvas_with_smear(source_image: Image.Image, dimensions: _Dimensions, horizontal_smear: bool = True) -> Image.Image:
    """ Creates a new canvas and primes it for outpainting by smearing the edges of the source image 

    For vertical expansion (horizontal_smear=False):
        - Uses source width but working height
        - Only expands vertically
    For horizontal expansion (horizontal_smear=True):
        - Uses full working width and height
        - Expands horizontally
    """
    if horizontal_smear:
        # For horizontal expansion:
        # - Use working width to allow horizontal expansion
        # - Keep current height from vertical expansion
        # - source_image is now the vertically expanded image
        canvas_width = dimensions.working.width
        canvas_height = source_image.height  # Keep the height we already have
        h_expansion = (dimensions.working.width - source_image.width) // 2
        v_expansion = 0  # No vertical expansion in horizontal phase
    else:
        # For vertical expansion:
        # - Keep source width (no horizontal change)
        # - Use working height to allow vertical expansion
        canvas_width = dimensions.source.width
        canvas_height = dimensions.working.height
        h_expansion = 0
        v_expansion = (dimensions.working.height - dimensions.source.height) // 2

    _logger.info("Creating and priming the canvas for outpainting")
    _logger.debug("Canvas dimensions: %dx%d", canvas_width, canvas_height)

    # Create a new canvas and paste the source image into the center
    final_canvas = Image.new("RGB", (canvas_width, canvas_height))
    final_canvas.paste(source_image, (h_expansion, v_expansion))

    if horizontal_smear:
        # Only smear horizontally if we're in horizontal mode
        # Smear the right edge
        right_edge = source_image.crop(
            (source_image.width - 1, 0, source_image.width, source_image.height))
        smeared_right = right_edge.resize(
            (h_expansion, source_image.height), Image.Resampling.BOX)
        final_canvas.paste(smeared_right, (source_image.width + h_expansion, 0))

        # Smear the left edge
        left_edge = source_image.crop((0, 0, 1, source_image.height))
        smeared_left = left_edge.resize(
            (h_expansion, source_image.height), Image.Resampling.BOX)
        final_canvas.paste(smeared_left, (0, 0))
    else:
        # Only smear vertically in vertical mode
        # Smear the bottom edge
        bottom_edge = source_image.crop(
            (0, dimensions.source.height - 1, dimensions.source.width, dimensions.source.height))
        smeared_bottom = bottom_edge.resize(
            (dimensions.source.width, v_expansion), Image.Resampling.BOX)
        final_canvas.paste(smeared_bottom, (0, dimensions.source.height + v_expansion))

        # Smear the top edge
        top_edge = source_image.crop((0, 0, dimensions.source.width, 1))
        smeared_top = top_edge.resize(
            (dimensions.source.width, v_expansion), Image.Resampling.BOX)
        final_canvas.paste(smeared_top, (0, 0))

    # Save the interim result with a direction-specific name
    canvas_type = "horizontal_canvas" if horizontal_smear else "vertical_canvas"
    _save_interim_result(final_canvas, canvas_type)

    return final_canvas


def _outpaint_side(side: _Side, dimensions: _Dimensions, generator: torch.Generator, pipeline: StableDiffusionXLInpaintPipeline, canvas: Image.Image) -> Image.Image:
    """ Generates an out painted side for the image """

    is_horizontal = side in [_Side.LEFT, _Side.RIGHT]

    # Calculate dimensions based on direction
    canvas_width = dimensions.working.width if is_horizontal else dimensions.source.width
    canvas_height = dimensions.working.height

    if is_horizontal:
        working_expansion = (dimensions.working.width - dimensions.source.width) // 2
        feather_size = max(1, int(dimensions.source.width * (_config.data.generation.outpaint.feathering / 100.0)))
        mask_size = working_expansion + feather_size
        mask_area = Image.new("L", (mask_size, canvas_height), 255)
    else:
        working_expansion = (dimensions.working.height - dimensions.source.height) // 2
        feather_size = max(1, int(dimensions.source.height * (_config.data.generation.outpaint.feathering / 100.0)))
        mask_size = working_expansion + feather_size
        mask_area = Image.new("L", (canvas_width, mask_size), 255)

    # Create the base mask (black = keep, white = paint)
    mask = Image.new("L", (canvas_width, canvas_height), 0)

    _logger.debug("Creating mask for %s outpainting: %dx%d",
                  "horizontal" if is_horizontal else "vertical",
                  canvas_width, canvas_height)

    # Apply feathering based on side
    if side == _Side.LEFT:
        for x in range(feather_size):
            x_pos = mask_size - feather_size + x
            alpha = int(255 * (1.0 - (x / feather_size)))
            for y in range(canvas_height):
                mask_area.putpixel((x_pos, y), alpha)
        mask_offset = (0, 0)
    elif side == _Side.RIGHT:
        for x in range(feather_size):
            alpha = int(255 * (x / feather_size))
            for y in range(canvas_height):
                mask_area.putpixel((x, y), alpha)
        mask_offset = (canvas_width - mask_size, 0)
    elif side == _Side.TOP:
        for y in range(feather_size):
            y_pos = mask_size - feather_size + y
            alpha = int(255 * (1.0 - (y / feather_size)))
            for x in range(canvas_width):
                mask_area.putpixel((x, y_pos), alpha)
        mask_offset = (0, 0)
    else:  # BOTTOM
        for y in range(feather_size):
            alpha = int(255 * (y / feather_size))
            for x in range(canvas_width):
                mask_area.putpixel((x, y), alpha)
        mask_offset = (0, canvas_height - mask_size)

    # Paste the feathered white area onto the original canvas
    mask.paste(mask_area, mask_offset)
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
            height=canvas_height,
            width=canvas_width
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
    source_image = Image.open(source_image_path).convert("RGB")
    source_width, source_height = source_image.size
    dimensions = _calculate_target_dimensions(source_width, source_height)

    _logger.header("Creating widened image")
    _logger.debug("The following properties will be used as part of the outpainting process")
    _logger.debug("  Source image: %dx%d", dimensions.source.width, dimensions.source.height)
    _logger.debug("  Target image: %dx%d", dimensions.target.width, dimensions.target.height)
    _logger.debug("  Working image: %dx%d", dimensions.working.width, dimensions.working.height)
    _logger.debug("  Feather size: %.1f%%", _config.data.generation.outpaint.feathering)
    _logger.debug("  Steps: %d", _config.data.generation.outpaint.steps)
    _logger.debug("  Guidance: %.1f", _config.data.generation.outpaint.guidance)

    generator = sdxl.create_generator(device, _config.data.generation.seed)

    # First handle vertical expansion
    _logger.info("Setting up vertical expansion canvas")
    working_canvas = _prime_canvas_with_smear(source_image, dimensions, horizontal_smear=False)

    _logger.info("Outpainting top side")
    working_canvas = _outpaint_side(_Side.TOP, dimensions, generator, pipe, working_canvas)

    _logger.info("Outpainting bottom side")
    working_canvas = _outpaint_side(_Side.BOTTOM, dimensions, generator, pipe, working_canvas)

    # Now handle horizontal expansion using the vertically expanded image
    _logger.info("Setting up horizontal expansion canvas")
    working_canvas = _prime_canvas_with_smear(working_canvas, dimensions, horizontal_smear=True)

    _logger.info("Outpainting right side")
    working_canvas = _outpaint_side(_Side.RIGHT, dimensions, generator, pipe, working_canvas)

    _logger.info("Outpainting left side")
    working_canvas = _outpaint_side(_Side.LEFT, dimensions, generator, pipe, working_canvas)

    # Crop to final target size to remove any edge artifacts
    crop_left = (dimensions.working.width - dimensions.target.width) // 2
    crop_top = (dimensions.working.height - dimensions.target.height) // 2
    crop_right = crop_left + dimensions.target.width
    crop_bottom = crop_top + dimensions.target.height

    final_image = working_canvas.crop((crop_left, crop_top, crop_right, crop_bottom))

    # Save the final result
    final_output_path = (
        Path(_args.output) /
        _config.data.paths.result_dir /
        _config.data.paths.temp_dir /
        _config.data.paths.outputs.stage_02
    )
    final_image.save(final_output_path, quality=_config.data.generation.outpaint.save_quality)

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
