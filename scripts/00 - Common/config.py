#!/usr/bin/env python3
""" Loads the root config.json file and provides easy access to configuration values """

import time

from pydantic import Field
from pydantic_utils import StrictBaseModel, JsonFileLoader, create_loader_function

# Pydantic models for structured configuration


class PromptsConfig(StrictBaseModel):
    """ Configuration for image generation prompts """
    image_positive: str = Field(..., min_length=1, description="Positive prompt for image generation")
    image_negative: str = Field("", description="Negative prompt for image generation")


class ImageDimensionsConfig(StrictBaseModel):
    """ Configuration for image dimensions """
    width: int = Field(1024, ge=1024, le=1024, description="Image width in pixels")
    height: int = Field(1024, ge=1024, le=1024, description="Image height in pixels")


class DimensionsConfig(StrictBaseModel):
    """ Configuration for various dimension settings """
    image: ImageDimensionsConfig = ImageDimensionsConfig()


class ImageGenerationConfig(StrictBaseModel):
    """ Configuration for image generation parameters """
    steps: int = Field(50, ge=1, le=200, description="Number of inference steps")
    base_fractal: float = Field(0.8, ge=0.1, le=0.9, description="Fraction of steps for base model")
    guidance: float = Field(6.0, ge=0, le=50, description="Guidance scale for generation")


class OutpaintConfig(StrictBaseModel):
    """ Configuration for outpainting parameters """
    steps: int = Field(50, ge=1, le=200, description="Number of inference steps for outpainting")
    guidance: float = Field(6.0, ge=0, le=50, description="Guidance scale for outpainting")
    feathering: float = Field(6.0, ge=0, le=10, description="Feathering percentage at edges")


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
    prompts: PromptsConfig
    dimensions: DimensionsConfig = DimensionsConfig()
    generation: GenerationConfig = GenerationConfig()
    paths: PathsConfig = PathsConfig()

    class Config:
        """ Pydantic configuration for ConfigData model """
        # Allow field aliases (like "01" -> "stage_01")
        populate_by_name = True


class Config(JsonFileLoader):
    """ Configuration object that loads and provides access to config.json values """

    def __init__(self, config_path: str = "../../config.json"):
        """ Initialize the config by loading from the specified path """
        super().__init__(config_path, ConfigData)

    @property
    def data(self) -> ConfigData:
        """ Get read-only access to the structured configuration data """
        return self._data


# Create the convenience function using the factory
load_config = create_loader_function(Config, "../../config.json")
