""" Common argument parser for all scripts in the project """
import argparse


class BaseArgumentParser:
    """ Base argument parser with common arguments for all scripts """

    def __init__(self, description: str | None = None):
        """ Initialize the argument parser with a description """
        self.parser = argparse.ArgumentParser(
            description=description or 'Script with common arguments'
        )
        self.add_base_arguments()

    def add_base_arguments(self) -> None:
        """ Add arguments that every script needs """
        self.parser.add_argument(
            '--config', '-c',
            type=str,
            help='Path to the configuration file containing the build properties',
            required=True
        )
        self.parser.add_argument(
            '--authentication', '-a',
            type=str,
            help='Path to the authentication file containing our service tokens',
            required=True
        )
        self.parser.add_argument(
            '--output', '-o',
            type=str,
            help='Output folder used to store all temp and generated content',
            required=True
        )
        self.parser.add_argument(
            '--log-file',
            type=str,
            default='pipeline.log',
            help='Log file path (default: pipeline.log)'
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
