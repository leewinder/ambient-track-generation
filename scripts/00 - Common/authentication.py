#!/usr/bin/env python3
""" Loads the root authentication.json file and provides easy access to authentication values """

import json
from pathlib import Path
from typing import Any

import jsonschema


# Strict schema definition - only allows defined authentication providers
AUTHENTICATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["huggingface"],
    "properties": {
        "huggingface": {
            "type": "string",
            "minLength": 1,
            "pattern": "^hf_[a-zA-Z0-9]{30,40}$"
        }
    }
}


class Authentication:
    """ Authentication object that loads and provides access to authentication.json values """

    def __init__(self, authentication_path: str = "../../authentication.json"):
        """ Initialize the authentication by loading from the specified path """
        self.authentication_path = Path(__file__).parent / authentication_path
        self._data = self._load_authentication()

    def _load_authentication(self) -> dict[str, Any]:
        """ Load and validate the authentication from the JSON file """
        try:
            with open(self.authentication_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Authentication file not found: {self.authentication_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in authentication file: {exc}") from exc

        # Validate against schema
        try:
            jsonschema.validate(data, AUTHENTICATION_SCHEMA)
        except jsonschema.ValidationError as exc:
            # Create helpful error message
            error_path = " -> ".join(str(p) for p in exc.absolute_path) if exc.absolute_path else "root"
            raise ValueError(
                f"Authentication validation error at '{error_path}': {exc.message}"
            ) from exc
        except jsonschema.SchemaError as exc:
            raise ValueError(f"Internal authentication schema error: {exc}") from exc

        return data

    # Properties
    @property
    def huggingface_token(self) -> str:
        """ Get the Hugging Face authentication token """
        return self._data.get("huggingface", "")


def load_authentication(authentication_path: str = "../../authentication.json") -> Authentication:
    """ Convenience function to load the authentication file """
    return Authentication(authentication_path)
