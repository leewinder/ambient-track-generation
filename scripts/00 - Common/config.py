#!/usr/bin/env python3
""" Loads the root config.json file and provides easy access to configuration values """

import json
import time
from pathlib import Path
from typing import Any

import jsonschema

# Strict schema definition - only allows defined fields, catches typos
CONFIG_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["prompts"],
    "properties": {
        "debug": {
            "type": "boolean"
        },
        "prompts": {
            "type": "object",
            "additionalProperties": False,
            "required": ["image_positive", "image_negative"],
            "properties": {
                "image_positive": {"type": "string", "minLength": 1},
                "image_negative": {"type": "string", "minLength": 1}
            }
        },
        "dimensions": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "image": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "width": {"type": "integer", "minimum": 64, "maximum": 4096},
                        "height": {"type": "integer", "minimum": 64, "maximum": 4096}
                    }
                }
            }
        },
        "generation": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "seed": {"type": "integer", "minimum": 0},
                "image": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "steps": {"type": "integer", "minimum": 1, "maximum": 100},
                        "base_fractal": {"type": "number", "minimum": 0.1, "maximum": 0.9},
                        "guidance": {"type": "number", "minimum": 0, "maximum": 20},
                    }
                },
                "outpaint": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "steps": {"type": "integer", "minimum": 1, "maximum": 100},
                        "guidance": {"type": "number", "minimum": 0, "maximum": 20},
                        "feathering": {"type": "number", "minimum": 0, "maximum": 10}
                    }
                }
            }
        },
        "paths": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "result_dir": {"type": "string", "minLength": 1},
                "temp_dir": {"type": "string", "minLength": 1},
                "outputs": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "01": {"type": "string", "minLength": 1},
                        "02": {"type": "string", "minLength": 1}
                    }
                }
            }
        }
    }
}


class Config:
    """ Configuration object that loads and provides access to config.json values """

    def __init__(self, config_path: str = "../../config.json"):
        """ Initialize the config by loading from the specified path """
        self.config_path = Path(__file__).parent / config_path
        self._data = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """ Load and validate the configuration from the JSON file """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in config file: {exc}") from exc

        # Validate against schema
        try:
            jsonschema.validate(data, CONFIG_SCHEMA)
        except jsonschema.ValidationError as exc:
            # Create helpful error message
            error_path = " -> ".join(str(p) for p in exc.absolute_path) if exc.absolute_path else "root"
            raise ValueError(
                f"Config validation error at '{error_path}': {exc.message}"
            ) from exc
        except jsonschema.SchemaError as exc:
            raise ValueError(f"Internal schema error: {exc}") from exc

        return data

    # --- General properties ---
    @property
    def debug(self) -> bool:
        """ Get debug mode state """
        return self._data.get("debug", False)

    # --- Image configuration ---
    @property
    def image_width(self) -> int:
        """ Get the image width from config """
        return self._data.get("dimensions", {}).get("image", {}).get("width", 1024)

    @property
    def image_height(self) -> int:
        """ Get the image height from config """
        return self._data.get("dimensions", {}).get("image", {}).get("height", 1024)

    # --- General generation configuration ---
    @property
    def generation_seed(self) -> int:
        """ Get the random seed for generation """
        return self._data.get("generation", {}).get("seed", int(time.time()))

    # --- Image generation configuration ---
    @property
    def image_generation_steps(self) -> int:
        """ Get the number of inference steps for image generation """
        return self._data.get("generation", {}).get("image", {}).get("steps", 50)

    @property
    def image_base_fractal(self) -> float:
        """ Get the fraction of steps the base model should handle """
        return self._data.get("generation", {}).get("image", {}).get("base_fractal", 0.8)

    @property
    def image_guidance(self) -> float:
        """ Get base guidance value the model should use """
        return self._data.get("generation", {}).get("image", {}).get("guidance", 6)

    # --- Outpainting generation configuration ---
    @property
    def outpaint_generation_steps(self) -> int:
        """ Get the number of inference steps for image generation """
        return self._data.get("generation", {}).get("outpaint", {}).get("steps", 50)

    @property
    def outpaint_guidance(self) -> float:
        """ Get base guidance value the model should use """
        return self._data.get("generation", {}).get("outpaint", {}).get("guidance", 6)

    @property
    def outpaint_feathering(self) -> float:
        """ Get the % of feathering at the edges the model should use """
        return self._data.get("generation", {}).get("outpaint", {}).get("feathering", 6)

    # --- Prompt access ---
    @property
    def image_prompt_positive(self) -> str:
        """ Get the first image prompt from config """
        return self._data.get("prompts", {}).get("image_positive", "")

    @property
    def image_prompt_negative(self) -> str:
        """ Get the first image negative prompt from config """
        return self._data.get("prompts", {}).get("image_negative", "")

    # --- Paths ---
    @property
    def result_dir(self) -> str:
        """ Get the output directory path """
        return self._data.get("paths", {}).get("result_dir", "result")

    @property
    def temp_dir(self) -> str:
        """ Get the temp output directory path """
        return self._data.get("paths", {}).get("temp_dir", "temp")

    # --- Output names ---
    @property
    def output_stage_01(self) -> str:
        """ Get the filename for stage 01 initial image output """
        return self._data.get("paths", {}).get("outputs", {}).get("01", "01_initial_image.png")

    @property
    def output_stage_02(self) -> str:
        """ Get the filename for stage 02 outpainted image output """
        return self._data.get("paths", {}).get("outputs", {}).get("02", "02_widened_image.png")


def load_config(config_path: str = "../../config.json") -> Config:
    """ Convenience function to load the configuration """
    return Config(config_path)
