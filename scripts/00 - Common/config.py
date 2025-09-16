#!/usr/bin/env python3
"""
Configuration loader for the AI media generation project.

Loads the root config.json file and provides easy access to configuration values.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration object that loads and provides access to config.json values."""

    def __init__(self, config_path: str = "../../config.json"):
        """
        Initialize the config by loading from the specified path.

        Args:
            config_path: Path to the config.json file, relative to this script
        """
        self.config_path = Path(__file__).parent / config_path
        self._data = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load the configuration from the JSON file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in config file: {exc}") from exc

    # --- Image configuration ---
    @property
    def image_width(self) -> int:
        """Get the image width from config."""
        return self._data.get("dimensions", {}).get("image", {}).get("width", 1024)

    @property
    def image_height(self) -> int:
        """Get the image height from config."""
        return self._data.get("dimensions", {}).get("image", {}).get("height", 1024)

    # --- Generation configuration ---
    @property
    def generation_seed(self) -> int:
        """Get the random seed for generation."""
        return self._data.get("generation", {}).get("seed", int(time.time()))

    @property
    def image_generation_steps(self) -> int:
        """Get the number of inference steps for image generation."""
        return self._data.get("generation", {}).get("image", {}).get("steps", 50)

    @property
    def image_base_fractal(self) -> float:
        """Get the fraction of steps the base model should handle."""
        return self._data.get("generation", {}).get("image", {}).get("base_fractal", 0.8)

    @property
    def image_guidance(self) -> float:
        """Get base guidance value the model should use."""
        return self._data.get("generation", {}).get("image", {}).get("guidance", 6)

    # --- Prompt access ---
    @property
    def image_prompt_positive(self) -> str:
        """Get the first image prompt from config."""
        return self._data.get("prompts", {}).get("image_positive", "")

    @property
    def image_prompt_negative(self) -> str:
        """Get the first image negative prompt from config."""
        return self._data.get("prompts", {}).get("image_negative", "")

    # --- Paths ---
    @property
    def result_dir(self) -> str:
        """Get the output directory path."""
        return self._data.get("paths", {}).get("result_dir", "result")

    @property
    def temp_dir(self) -> str:
        """Get the temp output directory path."""
        return self._data.get("paths", {}).get("temp_dir", "temp")

    @property
    def output_stage_01(self) -> str:
        """Get the output content."""
        return self._data.get("paths", {}).get("outputs", {}).get("01", "01_initial_image.png")


def load_config(config_path: str = "../../config.json") -> Config:
    """
    Convenience function to load the configuration.

    Args:
        config_path: Path to the config.json file

    Returns:
        Config object with loaded configuration
    """
    return Config(config_path)
