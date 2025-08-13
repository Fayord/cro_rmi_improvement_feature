"""
Centralized settings loaded from environment variables and optional .env file.

This module exposes typed constants for use across the project. It prioritizes
the env var NAME `NUMBER_OF_WORKER` (singular, as requested) with a fallback to
`NUMBER_OF_WORKERS` (plural) and finally a safe default of 32.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv, find_dotenv


# Load environment variables from the nearest .env file if present.
# find_dotenv() walks upward from CWD to locate a .env file.
load_dotenv(find_dotenv(), override=False)


def _get_int_env(name: str, fallback: Optional[int] = None) -> Optional[int]:
    value = os.getenv(name)
    if value is None:
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


# Thread/concurrency workers used across the app
NUMBER_OF_WORKERS: int = _get_int_env(
    "NUMBER_OF_WORKER",
    _get_int_env("NUMBER_OF_WORKERS", 32),
)
