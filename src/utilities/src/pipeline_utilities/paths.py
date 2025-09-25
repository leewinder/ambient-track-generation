""" Common paths used throughout the project """
import os
from typing import Final


class Paths:
    """ Defines the path properties that are used throughout the generation process """
    RESULT: Final[str] = "result"
    TEMP: Final[str] = "temp"

    OUTPUT_01: Final[str] = "01_initial_image.png"
    OUTPUT_02: Final[str] = "02_widened_image.png"
    OUTPUT_03: Final[str] = "03_upscaled_image.png"


def interim_save_folder(index: int, script_name: str) -> str:
    """ Returns the name of the interim folder for save data for a script"""
    return f"{index:02}_{os.path.splitext(os.path.basename(script_name))[0]}"
