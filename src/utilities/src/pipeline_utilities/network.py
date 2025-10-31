#!/usr/bin/env python3
""" Network utilities with retry logic for API calls """

from __future__ import annotations

import json
import logging
import time
from typing import Callable, Final

import requests

# Public API - functions and classes that external scripts should use
__all__ = [
    'NetworkRetry'
]


class NetworkRetry:
    """ Network request retry utilities for handling transient failures """

    MAX_RETRIES: Final[int] = 5
    INITIAL_BACKOFF_SECONDS: Final[float] = 1.0
    BACKOFF_MULTIPLIER: Final[float] = 2.0

    @staticmethod
    def _is_retryable_error(exception: Exception) -> bool:
        """ Check if an exception should trigger a retry """
        if isinstance(exception, (
            requests.exceptions.HTTPError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException
        )):
            return True
        return False

    @staticmethod
    def execute(
        request_func: Callable[[], requests.Response],
        max_retries: int = 5,
        initial_backoff: float = 1.0,
        backoff_multiplier: float = 2.0
    ) -> requests.Response:
        """ Execute a request function with exponential backoff """
        logger = logging.getLogger(__name__)
        last_exception: Exception | None = None
        backoff = initial_backoff

        for attempt in range(max_retries + 1):
            try:
                logger.debug("NetworkRetry.execute: Attempt %s/%s", attempt + 1, max_retries + 1)
                response = request_func()
                logger.debug("NetworkRetry.execute: Response received - Status: %s, URL: %s",
                             response.status_code, response.url)

                # Capture error response body before raising exception
                if not response.ok:
                    logger.error("=" * 80)
                    logger.error("NetworkRetry.execute: HTTP Error on attempt %s/%s", attempt + 1, max_retries + 1)
                    logger.error("  Status Code: %s", response.status_code)
                    logger.error("  URL: %s", response.url)
                    logger.error("  Response Headers: %s", json.dumps(dict(response.headers), indent=2))
                    try:
                        error_body = response.json()
                        logger.error("  Error Response Body: %s", json.dumps(error_body, indent=2))
                    except (ValueError, json.JSONDecodeError):
                        logger.error("  Error Response Body (raw, first 1000 chars): %s", response.text[:1000])
                    logger.error("=" * 80)

                response.raise_for_status()
                logger.debug("NetworkRetry.execute: Request succeeded on attempt %s", attempt + 1)
                return response
            except requests.exceptions.HTTPError as exc:
                last_exception = exc
                # Response body already logged above, but log exception details if response missing
                if not exc.response:
                    logger.error("=" * 80)
                    logger.error("NetworkRetry.execute: HTTPError on attempt %s/%s (no response object)",
                                 attempt + 1, max_retries + 1)
                    logger.error("  Exception: %s", exc)
                    logger.error("=" * 80)

                if attempt < max_retries:
                    wait_time = backoff * (backoff_multiplier ** attempt)
                    logger.debug("NetworkRetry.execute: Retrying in %.2f seconds...", wait_time)
                    time.sleep(wait_time)
                    continue
                logger.error("NetworkRetry.execute: All %s attempts exhausted, raising exception", max_retries + 1)
                raise
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                last_exception = exc
                logger.error("NetworkRetry.execute: %s on attempt %s/%s: %s",
                             type(exc).__name__, attempt + 1, max_retries + 1, exc)
                if attempt < max_retries:
                    wait_time = backoff * (backoff_multiplier ** attempt)
                    logger.debug("NetworkRetry.execute: Retrying in %.2f seconds...", wait_time)
                    time.sleep(wait_time)
                    continue
                raise
            except requests.exceptions.RequestException as exc:
                last_exception = exc
                logger.error("NetworkRetry.execute: RequestException on attempt %s/%s: %s",
                             attempt + 1, max_retries + 1, exc)
                if attempt < max_retries and NetworkRetry._is_retryable_error(exc):
                    wait_time = backoff * (backoff_multiplier ** attempt)
                    logger.debug("NetworkRetry.execute: Retrying in %.2f seconds...", wait_time)
                    time.sleep(wait_time)
                    continue
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Retry logic failed unexpectedly")
