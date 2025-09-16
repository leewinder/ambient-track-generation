"""
Common argument parser for all scripts in the project.
"""
import argparse


def parse_common_arguments(description: str = None) -> argparse.Namespace:
    """Parse common command line arguments for all scripts."""
    parser = argparse.ArgumentParser(description=description or 'Script with common arguments')

    # Common arguments for all scripts
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output folder used to store all temp and generated content',
        required=True
    )
    return parser.parse_args()
