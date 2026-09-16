"""What every parser here shares: one field check, and the one error the CLI reports rather than raises."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ConfigError(Exception):
    """The run as described cannot be run, whether it was described by a file or by a flag.

    This is the CLI's error contract: `uni` prints the message and exits EX_CONFIG. Anything
    else reaching it is a bug here, and a bug reaches the user as a traceback.
    """


def field(raw: Mapping[str, Any], key: str, kind: type, error: type[Exception]) -> Any:
    if key not in raw:
        raise error(f"{key} is missing")
    # Exact type, so a JSON or TOML boolean is not taken for an integer.
    if type(raw[key]) is not kind:
        raise error(f"{key} must be a {kind.__name__}, got {raw[key]!r}")
    return raw[key]


def nullable(raw: Mapping[str, Any], key: str, kind: type, error: type[Exception]) -> Any:
    """The field, which must be there and may be null: absent is a file that lost it, null is a recorded nothing."""
    if key in raw and raw[key] is None:
        return None
    return field(raw, key, kind, error)
