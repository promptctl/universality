"""The pinned model configuration, parsed once from pinned.toml."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Literal, get_args

Dtype = Literal["float32", "float16", "bfloat16"]
Device = Literal["mps", "cpu", "cuda"]

COMMIT = re.compile(r"[0-9a-f]{40}")


class PinnedConfigError(Exception):
    """pinned.toml does not pin a model. The message says which field is wrong."""


@dataclass(frozen=True)
class Pinned:
    model_id: str
    revision: str  # a full commit sha on the model's hub repo
    dtype: Dtype
    device: Device
    max_new_tokens: int


def _field(raw: dict[str, Any], path: str, kind: type) -> Any:
    table, key = path.split(".")
    section = raw.get(table)
    if not (isinstance(section, dict) and key in section):
        raise PinnedConfigError(f"{path} is missing")
    value = section[key]
    # Exact type, so a TOML boolean is not taken for an integer.
    if type(value) is not kind:
        raise PinnedConfigError(f"{path} must be a {kind.__name__}, got {value!r}")
    return value


def _choice(raw: dict[str, Any], path: str, choices: tuple[str, ...]) -> Any:
    value = _field(raw, path, str)
    if value not in choices:
        raise PinnedConfigError(f"{path} must be one of {', '.join(choices)}, got {value!r}")
    return value


def parse_pinned(text: str) -> Pinned:
    # [LAW:parse-dont-validate] a branch-name revision or a typo'd device stops here, not after a download.
    raw = tomllib.loads(text)
    revision = _field(raw, "model.revision", str)
    if not COMMIT.fullmatch(revision):
        raise PinnedConfigError(f"model.revision must be a full commit sha, got {revision!r}")
    max_new_tokens = _field(raw, "generation.max_new_tokens", int)
    if max_new_tokens <= 0:
        raise PinnedConfigError(f"generation.max_new_tokens must be positive, got {max_new_tokens}")
    return Pinned(
        model_id=_field(raw, "model.id", str),
        revision=revision,
        dtype=_choice(raw, "model.dtype", get_args(Dtype)),
        device=_choice(raw, "model.device", get_args(Device)),
        max_new_tokens=max_new_tokens,
    )


def load_pinned() -> Pinned:
    return parse_pinned(files("uni").joinpath("pinned.toml").read_text())
