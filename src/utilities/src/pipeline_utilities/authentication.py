#!/usr/bin/env python3
""" Loads the root authentication.json file and provides easy access to authentication values """

import re

from pydantic import Field, field_validator
from .pydantic_utils import StrictBaseModel, JsonFileLoader, create_loader_function
from . import logging_utils

# Regex pattern for Hugging Face token validation
HF_TOKEN_PATTERN = r'^hf_[a-zA-Z0-9]{30,40}$'


class AuthenticationData(StrictBaseModel):
    """ Pydantic model for authentication data with validation """

    huggingface: str = Field(
        ...,
        min_length=1,
        description="Hugging Face authentication token"
    )

    @field_validator('huggingface')
    @classmethod
    def validate_huggingface_token(cls, v: str) -> str:
        """ Validate Hugging Face token format """
        if not re.match(HF_TOKEN_PATTERN, v):
            raise ValueError('Invalid Hugging Face token format. Must be hf_ followed by 30-40 alphanumeric characters')
        return v


class Authentication(JsonFileLoader):
    """ Authentication object that loads and provides access to authentication.json values """

    def __init__(self, authentication_path: str):
        """ Initialize the authentication by loading from the specified path """
        logger = logging_utils.get_logger(__name__)
        logger.debug("Loading authentication from: %s", authentication_path)
        super().__init__(authentication_path, AuthenticationData)

    @property
    def data(self) -> AuthenticationData:
        """ Get read-only access to the structured authentication data """
        return self._data


# Create the convenience function using the factory
load_authentication_config = create_loader_function(Authentication, "authentication.json")
