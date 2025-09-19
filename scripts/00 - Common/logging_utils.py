#!/usr/bin/env python3
""" Common logging setup for all pipeline scripts """

import logging
import sys
from pathlib import Path


def setup_pipeline_logging(
    log_file: str = "pipeline.log",
    debug: bool = False,
    script_name: str | None = None
) -> logging.Logger:
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

    # Log that this script started
    logger = logging.getLogger(script_name or __name__)
    logger.info("=== %s Started ===", script_name or 'Script')
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """ Get a logger for the calling module """
    return logging.getLogger(name or __name__)
