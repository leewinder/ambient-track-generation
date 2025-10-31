#!/usr/bin/env python3
""" Authentication credentials management """

from typing import Optional

from pydantic import BaseModel, Field

from .pydantics import JsonFileLoader

# Public API - functions and classes that external scripts should use
__all__ = [
    'SunoAuthentication',
    'AuthenticationModel',
    'AuthenticationLoader',
    'load_authentication'
]


class SunoAuthentication(BaseModel):
    """ Suno authentication credentials """
    session: str = Field(description="Suno session cookie")
    client_uat: str = Field(description="Suno client UAT cookie")
    client: Optional[str] = Field(default=None, description="Suno client cookie (optional)")


class AuthenticationModel(BaseModel):
    """ Authentication model with typed fields for known services """

    model_config = {"extra": "allow"}

    suno: Optional[SunoAuthentication] = Field(default=None, description="Suno authentication credentials")

    def get(self, key: str, default: str = "") -> str:
        """ Get authentication value with default """
        data = self.model_dump()
        return data.get(key, default)


class AuthenticationLoader(JsonFileLoader):
    """ Loader for authentication files """

    def __init__(self, file_path: str):
        """ Initialize the authentication loader """
        super().__init__(file_path, AuthenticationModel)


# Convenience function for loading authentication files
load_authentication = JsonFileLoader.create_loader_function(AuthenticationLoader, "authentication.json")
