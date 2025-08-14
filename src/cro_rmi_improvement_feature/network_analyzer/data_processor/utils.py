from typing import Optional


def _parse_bool_env(raw_value: Optional[str], default: bool = True) -> bool:
    """Parse a boolean-like environment variable robustly.

    Accepts common truthy/falsy strings, optionally wrapped in quotes, and is
    case-insensitive. Falls back to the provided default when value is None.
    """
    if raw_value is None:
        return default

    # Normalize and strip surrounding quotes and whitespace
    normalized = str(raw_value).strip().strip('"').strip("'").lower()

    truthy = {"1", "true", "t", "yes", "y", "on"}
    falsy = {"0", "false", "f", "no", "n", "off"}

    if normalized in truthy:
        return True
    if normalized in falsy:
        return False

    raise ValueError(f"Unknown USE_CACHED_LLM: >{raw_value}<")
