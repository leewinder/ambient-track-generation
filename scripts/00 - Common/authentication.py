#!/usr/bin/env python3
"""
Authentication loader for the AI media generation project.

Loads the root authentication.json file and provides easy access to authentication values.
"""

import json
from pathlib import Path
from typing import Any, Dict


class Authentication:
    """Authentication object that loads and provides access to authentication.json values."""

    def __init__(self, authentication_path: str = "../../authentication.json"):
        """
        Initialize the authentication by loading from the specified path.

        Args:
            authentication_path: Path to the authentication.json file, relative to this script
        """
        self.authentication_path = Path(__file__).parent / authentication_path
        self._data = self._load_authentication()

    def _load_authentication(self) -> Dict[str, Any]:
        """Load the authentication from the JSON file."""
        try:
            with open(self.authentication_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Authentication file not found: {self.authentication_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in authentication file: {exc}") from exc

    # Properties
    @property
    def huggingface_token(self) -> str:
        """Get the Hugging Face authentication token."""
        return self._data.get("huggingface", "")


def load_authentication(authentication_path: str = "../../authentication.json") -> Authentication:
    """
    Convenience function to load the authentication file.

    Args:
        authentication_path: Path to the authentication.json file

    Returns:
        Authentication object with loaded authentication
    """
    return Authentication(authentication_path)
