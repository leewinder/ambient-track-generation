""" Common argument parser for all scripts in the project """
import argparse

# Public API - functions and classes that external scripts should use
__all__ = [
    'BaseArgumentParser',
    'parse_arguments'
]


class BaseArgumentParser:
    """ Base argument parser with common arguments for all scripts """

    def __init__(self, description: str | None = None):
        """ Initialize the argument parser with a description """
        self.parser = argparse.ArgumentParser(
            description=description or 'Script with common arguments'
        )
        self.add_base_arguments()

        # Automatically call add_additional_arguments if it exists in the derived class
        if hasattr(self, 'add_additional_arguments'):
            self.add_additional_arguments()

    def add_base_arguments(self) -> None:
        """ Add arguments that every script needs """
        self.parser.add_argument(
            '--log-file',
            type=str,
            default='pipeline.log',
            help='Log file path (default: pipeline.log)'
        )
        self.parser.add_argument(
            '--step',
            type=str,
            required=True,
            help='Step name to execute from configuration.json'
        )

    def parse(self) -> argparse.Namespace:
        """ Parse and return the command line arguments """
        return self.parser.parse_args()


# Flexible argument parsing function
def parse_arguments(
    description: str | None = None,
    arg_parser: type[BaseArgumentParser] = BaseArgumentParser
) -> argparse.Namespace:
    """ Parse arguments using the specified parser class """
    return arg_parser(description).parse()
