#!/usr/bin/env python3
""" Common Pydantic utilities for validation and error handling """

import json
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

# Public API - functions and classes that external scripts should use
__all__ = [
    'StrictBaseModel',
    'JsonFileLoader',
    'create_loader_function'
]


class StrictBaseModel(BaseModel):
    """ Base model with strict validation (no extra fields allowed) """
    class Config:
        """ Base config to ensure we cannot have additional properties """
        extra = "forbid"


class JsonFileLoader:
    """ Base class for loading and validating JSON files with Pydantic models """

    def __init__(self, file_path: str, model_class: type[StrictBaseModel]):
        """ Initialize the loader with file path and Pydantic model class """
        # Look for config files in the current working directory
        self.file_path = Path(file_path)
        self.model_class = model_class
        try:
            self._data = self._load_and_validate()
        except ValueError as exc:
            # Re-raise with clean traceback
            raise ValueError(str(exc)) from None

    @property
    def data(self) -> StrictBaseModel:
        """ Get read-only access to the structured data """
        return self._data

    def _load_and_validate(self) -> StrictBaseModel:
        """ Load and validate the JSON file using the Pydantic model """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"File not found: {self.file_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in file: {exc}") from exc

        # Validate and parse using Pydantic
        try:
            return self.model_class(**raw_data)
        except Exception as exc:
            # Convert Pydantic errors to user-friendly messages
            error_msg = self._format_validation_error(exc)
            file_type = self.__class__.__name__.replace('Loader', '').lower()
            raise ValueError(f"{file_type} validation error: {error_msg}") from None

    def _format_validation_error(self, exc: Exception) -> str:
        """ Convert Pydantic validation errors to user-friendly messages """
        if hasattr(exc, 'errors'):
            errors = []
            for error in exc.errors():
                field_path = " -> ".join(str(loc) for loc in error['loc'])
                error_type = error['type']

                if error_type == 'missing':
                    errors.append(f"Missing required field '{field_path}'")
                elif error_type == 'extra_forbidden':
                    errors.append(f"Unknown field '{field_path}' (typo?)")
                elif error_type == 'value_error':
                    errors.append(f"Invalid value for '{field_path}': {error['msg']}")
                elif error_type in ['greater_than_equal', 'less_than_equal']:
                    errors.append(f"Value for '{field_path}' is out of allowed range: {error['msg']}")
                else:
                    errors.append(f"Error in '{field_path}': {error['msg']}")

            return "; ".join(errors)

        return str(exc)

    @staticmethod
    def create_loader_function(loader_class: type, default_path: str) -> Callable[[str], 'JsonFileLoader']:
        """ Factory function to create load_* convenience functions """
        def load_function(file_path: str = default_path) -> 'JsonFileLoader':
            try:
                return loader_class(file_path)
            except ValueError as exc:
                raise ValueError(str(exc)) from None

        return load_function
