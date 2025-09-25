#!/usr/bin/env python3
"""
Stable Diffusion Out Painter
Takes a pre-generated image and expands it to 1080p using a truly seamless 'single shot' inpainting method
"""

import gc
import sys
from typing import Final
from pathlib import Path
from dataclasses import dataclass
from PIL import Image, ImageFilter, ImageChops
import torch
from diffusers import StableDiffusionXLInpaintPipeline
import numpy

# Import our local utilities
from pipeline_utilities import authentication, generation, args, logging_utils, paths
from pipeline_sdxl import sdxl_utils as sdxl


class _DefaultProperties:
    OUTPUT_WIDTH: Final[int] = 1920
    OUTPUT_HEIGHT: Final[int] = 1080
    EXPANSION_PERCENT: Final[int] = 10
    NOISE_STRENGTH: Final[float] = 0.1
    FEATHERING: Final[float] = 5


@dataclass(frozen=True, slots=True)
class _DimensionPair:
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class _Dimensions:
    source: _DimensionPair
    target: _DimensionPair
    working: _DimensionPair


# --- Argument Parsing and Configuration ---
_args = args.parse_arguments("Expands a given image using Stable Diffusion's inpainting model")
_authentication = authentication.load_authentication_config(_args.authentication)
_config = generation.load_generation_config(_args.config)
_logger = logging_utils.setup_pipeline_logging(
    log_file=_args.log_file,
    debug=_config.data.debug
)


def _calculate_target_dimensions(source_width: int, source_height: int) -> _Dimensions:
    """ Calculate target and working dimensions with an oversized canvas to remove the edge artifacts if they happen """
    target_width = _DefaultProperties.OUTPUT_WIDTH
    target_height = _DefaultProperties.OUTPUT_HEIGHT
    expansion_width = int(target_width * (_DefaultProperties.EXPANSION_PERCENT / 100))
    expansion_height = int(target_height * (_DefaultProperties.EXPANSION_PERCENT / 100))

    # Create the working hight which is 1080p + the 10%, but also that it's a multiple of 8
    working_width = target_width + (2 * expansion_width)
    working_height = target_height + (2 * expansion_height)
    working_width = ((working_width + 7) // 8) * 8
    working_height = ((working_height + 7) // 8) * 8
    return _Dimensions(
        source=_DimensionPair(source_width, source_height),
        target=_DimensionPair(target_width, target_height),
        working=_DimensionPair(working_width, working_height)
    )


def _prime_full_canvas_with_smear(source_image: Image.Image, dimensions: _Dimensions) -> Image.Image:
    """ Creates a new canvas and primes it using a two-step smearing process, fills the corners, and adds noise. """

    _logger.info("Priming canvas with two-step smearing process")

    s_w, s_h = source_image.size
    h_expansion = (dimensions.working.width - s_w) // 2
    v_expansion = (dimensions.working.height - s_h) // 2

    # Step 1: Smear horizontally
    horizontal_canvas = Image.new("RGB", (dimensions.working.width, s_h))
    horizontal_canvas.paste(source_image, (h_expansion, 0))

    left_edge = source_image.crop((0, 0, 1, s_h)).resize((h_expansion, s_h), Image.Resampling.BOX)
    horizontal_canvas.paste(left_edge, (0, 0))

    right_edge = source_image.crop((s_w - 1, 0, s_w, s_h)).resize((h_expansion, s_h), Image.Resampling.BOX)
    horizontal_canvas.paste(right_edge, (s_w + h_expansion, 0))

    # Step 2: Smear vertically using the result of the horizontal smear
    final_canvas = Image.new("RGB", (dimensions.working.width, dimensions.working.height))
    final_canvas.paste(horizontal_canvas, (0, v_expansion))

    top_edge = horizontal_canvas.crop((0, 0, dimensions.working.width, 1)).resize(
        (dimensions.working.width, v_expansion), Image.Resampling.BOX)
    final_canvas.paste(top_edge, (0, 0))

    bottom_edge = horizontal_canvas.crop((0, s_h - 1, dimensions.working.width, s_h)
                                         ).resize((dimensions.working.width, v_expansion), Image.Resampling.BOX)
    final_canvas.paste(bottom_edge, (0, s_h + v_expansion))

    # Step 3: Fill all corner gaps with vertically flipped adjacent sections
    _logger.info("Filling corner gaps with vertically flipped adjacent content")
    if h_expansion > 0 and v_expansion > 0:
        # Top-left corner: Copy area below and flip vertically
        tl_source_box = (0, v_expansion, h_expansion, v_expansion * 2)
        source_tl = final_canvas.crop(tl_source_box)
        flipped_tl = source_tl.transpose(Image.Transpose.FLIP_TOP_BOTTOM)  # Vertical flip
        final_canvas.paste(flipped_tl, (0, 0))

        # Top-right corner: Copy area below and flip vertically
        tr_source_box = (h_expansion + s_w, v_expansion, dimensions.working.width, v_expansion * 2)
        source_tr = final_canvas.crop(tr_source_box)
        flipped_tr = source_tr.transpose(Image.Transpose.FLIP_TOP_BOTTOM)  # Vertical flip
        final_canvas.paste(flipped_tr, (h_expansion + s_w, 0))

        # Bottom-left corner: Copy area above and flip vertically
        bl_source_box = (0, v_expansion + s_h - v_expansion, h_expansion, v_expansion + s_h)
        source_bl = final_canvas.crop(bl_source_box)
        flipped_bl = source_bl.transpose(Image.Transpose.FLIP_TOP_BOTTOM)  # Vertical flip
        final_canvas.paste(flipped_bl, (0, v_expansion + s_h))

        # Bottom-right corner: Copy area above and flip vertically
        br_source_box = (h_expansion + s_w, v_expansion + s_h - v_expansion,
                         dimensions.working.width, v_expansion + s_h)
        source_br = final_canvas.crop(br_source_box)
        flipped_br = source_br.transpose(Image.Transpose.FLIP_TOP_BOTTOM)  # Vertical flip
        final_canvas.paste(flipped_br, (h_expansion + s_w, v_expansion + s_h))

    # --- Step 4: Add noise layer to the entire canvas ---
    _logger.info("Adding noise layer with strength: %s", _DefaultProperties.NOISE_STRENGTH)
    if _DefaultProperties.NOISE_STRENGTH > 0:
        # Generate monochromatic noise using numpy
        noise_array = numpy.random.randint(0, 256, (dimensions.working.height, dimensions.working.width), dtype='uint8')
        # Convert to a PIL Image that can be used with ImageChops
        noise_image = Image.fromarray(noise_array).convert("L").convert("RGB")

        # Create the fully noise-overlaid version
        overlaid_image = ImageChops.overlay(final_canvas, noise_image)

        # Blend the original canvas with the overlaid version to control strength
        final_canvas = Image.blend(final_canvas, overlaid_image, alpha=_DefaultProperties.NOISE_STRENGTH)

        # --- Step 5: Restore the clean source image in the center ---
        # This paste ensures the noise is only in the out painted areas.
        final_canvas.paste(source_image, (h_expansion, v_expansion))

    _save_interim_result(final_canvas, "01_primed_canvas")

    _clear_memory()
    return final_canvas


def _create_seamless_central_mask(dimensions: _Dimensions) -> Image.Image:
    """ Creates a perfectly smooth, feathered mask using Gaussian Blur """

    _logger.info("Creating seamless central feathered mask using Gaussian Blur")

    mask = Image.new("L", (dimensions.working.width, dimensions.working.height), 255)

    black_rect = Image.new("L", (dimensions.source.width, dimensions.source.height), 0)
    paste_x = (dimensions.working.width - dimensions.source.width) // 2
    paste_y = (dimensions.working.height - dimensions.source.height) // 2
    mask.paste(black_rect, (paste_x, paste_y))

    feather_radius = max(1, int(min(dimensions.source.width, dimensions.source.height)
                         * (_DefaultProperties.FEATHERING / 100.0) / 2))

    blurred_mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))

    _save_interim_result(blurred_mask, "02_central_mask")

    _clear_memory()
    return blurred_mask


def _run_outpainting_model(pipeline: StableDiffusionXLInpaintPipeline, canvas: Image.Image, mask: Image.Image, generator: torch.Generator) -> Image.Image:
    """ Runs a single inpainting pass with the given canvas and mask """
    with torch.no_grad():
        result = pipeline(
            prompt=_config.data.prompts.image_positive,
            negative_prompt=_config.data.prompts.image_negative,
            image=canvas,
            mask_image=mask,
            num_inference_steps=_config.data.generation.outpaint.steps,
            guidance_scale=_config.data.generation.outpaint.guidance,
            generator=generator,
            height=canvas.height,
            width=canvas.width
        ).images[0]
    _clear_memory()
    return result


def _save_interim_result(interim_image: Image.Image, name: str) -> None:
    """ Outputs interim images if debugging is enabled """
    if _config.data.debug:
        interim_path = (
            Path(_args.output) /
            paths.Paths.RESULT /
            paths.Paths.TEMP /
            paths.interim_save_folder(2, __file__) /
            f"{name}.png"
        )
        interim_path.parent.mkdir(parents=True, exist_ok=True)
        interim_image.save(interim_path)
        _logger.debug("Saved interim result: %s", interim_path)


def _clear_memory() -> None:
    """ Clears system and MPS memory """
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _outpaint_image() -> str:
    """ Generates a new out painted image from a source image using the single-shot method """
    source_image_path = (
        Path(_args.output) /
        paths.Paths.RESULT /
        paths.Paths.TEMP /
        paths.Paths.OUTPUT_01
    )
    if not source_image_path.is_file():
        raise FileNotFoundError(f"Required source image not found: {source_image_path}")

    _logger.header("Setting up Stable Diffusion Inpainting")
    device = sdxl.get_device()
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=sdxl.get_optimal_dtype(device),
        use_safetensors=True,
        token=_authentication.data.huggingface,
        add_watermarker=False
    ).to(device)
    sdxl.optimize_pipeline(pipe, device)

    source_image = Image.open(source_image_path).convert("RGB")
    dimensions = _calculate_target_dimensions(source_image.width, source_image.height)
    generator = sdxl.create_generator(device, _config.data.generation.seed)

    _logger.header("Starting Image Expansion")

    _logger.info("Generating primed canvas and mask")
    primed_canvas = _prime_full_canvas_with_smear(source_image, dimensions)
    mask = _create_seamless_central_mask(dimensions)

    _logger.info("Running single outpainting pass")
    final_canvas = _run_outpainting_model(pipe, primed_canvas, mask, generator)
    _save_interim_result(final_canvas, "03_expanded_result")

    pipe.to('cpu')
    del pipe
    _clear_memory()

    _logger.info("Cropping expanded image to 1080p")
    crop_box_final = (
        (dimensions.working.width - dimensions.target.width) // 2,
        (dimensions.working.height - dimensions.target.height) // 2,
        ((dimensions.working.width - dimensions.target.width) // 2) + dimensions.target.width,
        ((dimensions.working.height - dimensions.target.height) // 2) + dimensions.target.height
    )
    final_image = final_canvas.crop(crop_box_final)

    final_output_path = (
        Path(_args.output) /
        paths.Paths.RESULT /
        paths.Paths.TEMP /
        paths.Paths.OUTPUT_02
    )
    final_image.save(final_output_path)
    return str(final_output_path)


def main() -> None:
    """ Main entry point """
    try:
        output_path = _outpaint_image()
        _logger.info("Success! Image out painted: %s", output_path)
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        _logger.error("%s: %s", type(e).__name__, e)
        sys.exit(1)


if __name__ == "__main__":
    main()
