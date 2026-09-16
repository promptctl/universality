"""What every file parser here shares: one field check, and one error for a file that does not describe the run."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ConfigError(Exception):
    """A file that should describe the run does not. The message names the file and the field."""


def field(raw: Mapping[str, Any], key: str, kind: type, error: type[Exception]) -> Any:
    if key not in raw:
        raise error(f"{key} is missing")
    # Exact type, so a JSON or TOML boolean is not taken for an integer.
    if type(raw[key]) is not kind:
        raise error(f"{key} must be a {kind.__name__}, got {raw[key]!r}")
    return raw[key]
