#!/usr/bin/env python3
""" Loads the root generation.json file and provides easy access to configuration values """

import time

from pydantic import Field
from typing import Optional
from .pydantic_utils import StrictBaseModel, JsonFileLoader, create_loader_function
from . import logging_utils

# Public API - functions and classes that external scripts should use
__all__ = [
    'Generation',
    'load_generation_config'
]


# Pydantic models for structured generation
class Lora(StrictBaseModel):
    """ Configuration for a Lora model with ID and weight """
    lora: str = Field(..., min_length=1, description="Lora model identifier")
    weight: float = Field(..., description="Lora weight/strength")
    repo: str = Field("", description="Optional repository for the LoRA")


class PromptPair(StrictBaseModel):
    """ Generic prompt pair with positive and negative prompts - both are required """
    positive: str = Field(..., min_length=1, description="Positive prompt")
    negative: str = Field(..., description="Negative prompt")


class PromptsConfig(StrictBaseModel):
    """ Configuration for image generation prompts across different stages """
    image_base: PromptPair = Field(..., description="Base prompts for initial image generation")
    image_refiner: Optional[PromptPair] = Field(None, description="Optional refiner-specific prompts")
    image_outpaint: Optional[PromptPair] = Field(None, description="Optional outpaint-specific prompts")

    def get_base_image_prompts(self) -> PromptPair:
        """ Get base prompts for initial image generation """
        return self.image_base

    def get_refiner_image_prompts(self) -> PromptPair:
        """ Get refiner prompts, falling back to base prompts if refiner prompts are not provided """
        return self.image_refiner if self.image_refiner is not None else self.image_base

    def get_outpaint_image_prompts(self) -> PromptPair:
        """ Get outpaint prompts, falling back to base prompts if outpaint prompts are not provided """
        return self.image_outpaint if self.image_outpaint is not None else self.image_base


class ImageGenerationConfig(StrictBaseModel):
    """ Configuration for image generation parameters """
    steps: int = Field(50, ge=1, le=200, description="Number of inference steps")
    base_fractal: float = Field(0.8, ge=0.1, le=0.9, description="Fraction of steps for base model")
    guidance: float = Field(6.0, ge=0, le=50, description="Guidance scale for generation")
    base_checkpoints: str = Field("stabilityai/stable-diffusion-xl-base-1.0", description="Base model checkpoint")
    refiner_checkpoints: str = Field("stabilityai/stable-diffusion-xl-refiner-1.0",
                                     description="Refiner model checkpoint")
    base_loras: list[Lora] = Field(default_factory=list, description="Lora models for base generation")
    refiner_loras: list[Lora] = Field(default_factory=list, description="Lora models for refiner generation")


class OutpaintConfig(StrictBaseModel):
    """ Configuration for outpainting parameters """
    steps: int = Field(50, ge=1, le=200, description="Number of inference steps for outpainting")
    guidance: float = Field(6.0, ge=0, le=50, description="Guidance scale for outpainting")
    checkpoint: str = Field("diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                            description="Inpainting model checkpoint")
    loras: list[Lora] = Field(default_factory=list, description="Lora models for outpainting")


class GenerationConfig(StrictBaseModel):
    """ Configuration for generation parameters """
    seed: int = Field(default_factory=lambda: int(time.time()), ge=0, description="Random seed for generation")
    image: ImageGenerationConfig = ImageGenerationConfig()
    outpaint: OutpaintConfig = OutpaintConfig()


class OutputsConfig(StrictBaseModel):
    """ Configuration for output file names """
    stage_01: str = Field("01_initial_image.png", min_length=1, alias="01")
    stage_02: str = Field("02_widened_image.png", min_length=1, alias="02")


class PathsConfig(StrictBaseModel):
    """ Configuration for file paths """
    result_dir: str = Field("result", min_length=1, description="Directory for final results")
    temp_dir: str = Field("temp", min_length=1, description="Directory for temporary files")
    outputs: OutputsConfig = OutputsConfig()


class ConfigData(StrictBaseModel):
    """ Main configuration data model """
    debug: bool = Field(False, description="Enable debug mode")
    name: str = Field("", description="Name for this generation run")
    prompts: PromptsConfig
    generation: GenerationConfig = GenerationConfig()
    paths: PathsConfig = PathsConfig()

    class Config:
        """ Pydantic configuration for ConfigData model """
        # Allow field aliases (like "01" -> "stage_01")
        populate_by_name = True


class Generation(JsonFileLoader):
    """ Configuration object that loads and provides access to config.json values """

    def __init__(self, config_path: str):
        """ Initialize the config by loading from the specified path """
        logger = logging_utils.get_logger(__name__)
        logger.debug("Loading configuration from: %s", config_path)
        super().__init__(config_path, ConfigData)

    @property
    def data(self) -> ConfigData:
        """ Get read-only access to the structured configuration data """
        return self._data


# Create the convenience function using the factory
load_generation_config = create_loader_function(Generation, "generation.json")
