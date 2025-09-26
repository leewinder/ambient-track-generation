#!/usr/bin/env python3
""" Common logging setup for all pipeline scripts """

import logging
import sys
from pathlib import Path

# Public API - functions and classes that external scripts should use
__all__ = [
    'EnhancedLogger',
    'get_logger',
    'setup_pipeline_logging'
]


class EnhancedLogger:
    """ Enhanced logger with custom methods for better formatting """

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def header(self, text: str) -> None:
        """ Log a formatted header with dashes and uppercase text """
        # Log two empty lines first, then the header
        self._logger.info("")
        self._logger.info("")
        formatted_text = f"---- {text.upper()} ----"
        self._logger.info(formatted_text)

    def __getattr__(self, name):
        """ Delegate all other methods to the underlying logger """
        return getattr(self._logger, name)


def get_logger(name: str | None = None) -> EnhancedLogger:
    """ Get an enhanced logger for the calling module """
    base_logger = logging.getLogger(name or __name__)
    return EnhancedLogger(base_logger)


def setup_pipeline_logging(
    log_file: str = "pipeline.log",
    debug: bool = False,
    script_name: str | None = None
) -> "EnhancedLogger":
    """ Standard logging setup for all pipeline scripts """
    level = logging.DEBUG if debug else logging.INFO

    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, mode='a'),  # Append to shared file
            logging.StreamHandler(sys.stdout)         # Also show in console
        ],
        force=True  # Override any existing config
    )

    # Return enhanced logger for the script
    base_logger = logging.getLogger(script_name or __name__)
    return EnhancedLogger(base_logger)
