"""The one field check the file parsers share."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def field(raw: Mapping[str, Any], key: str, kind: type, error: type[Exception]) -> Any:
    if key not in raw:
        raise error(f"{key} is missing")
    # Exact type, so a JSON or TOML boolean is not taken for an integer.
    if type(raw[key]) is not kind:
        raise error(f"{key} must be a {kind.__name__}, got {raw[key]!r}")
    return raw[key]
