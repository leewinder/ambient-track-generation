""" Common paths used throughout the project """
from pathlib import Path
from typing import Final

# Public API - functions and classes that external scripts should use
__all__ = [
    'Project',
    'Paths'
]


class Project:
    """ Core project utilities """
    CONFIGURATION: Final[str] = "configuration.json"

    @staticmethod
    def get_root() -> Path:
        """ Traverse up the directory tree until we find a folder with configuration.json """
        current_path = Path(__file__).resolve()
        while current_path != current_path.parent:
            if (current_path / "configuration.json").exists():
                return current_path
            current_path = current_path.parent
        raise FileNotFoundError("Could not find project root (folder containing configuration.json)")

    @staticmethod
    def get_configuration() -> str:
        """ Get the full path to a configuration file as string """
        return str(Project.get_root() / Project.CONFIGURATION)

    @staticmethod
    def get_root_path(filename: str) -> str:
        """ Get the full path to a file in the project root """
        return str(Project.get_root() / filename)


class Paths:
    """ Defines the path properties that are used throughout the generation process """
    OUTPUT: Final[str] = "output"
    INTERIM: Final[str] = "interim"
    ARCHIVE: Final[str] = "archive"
    WORKFLOWS: Final[str] = "workflows"

    @staticmethod
    def get_interim_path(filename: str) -> Path:
        """ Get the full path to a working file in the working directory """
        return (
            Project.get_root() /
            Paths.OUTPUT /
            Paths.INTERIM /
            filename
        )

    @staticmethod
    def get_archive_path(filename: str) -> Path:
        """ Get the full path to a working file in the working directory """
        return (
            Project.get_root() /
            Paths.OUTPUT /
            Paths.ARCHIVE /
            filename
        )

    @staticmethod
    def get_workflows_path(filename: str) -> Path:
        """ Get the full path to a workflow file """
        return (
            Project.get_root() /
            Paths.WORKFLOWS /
            filename
        )
