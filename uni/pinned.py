"""The pinned model configuration, parsed once from pinned.toml."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from importlib.resources import files
from typing import Literal, get_args

Dtype = Literal["float32", "float16", "bfloat16"]

COMMIT = re.compile(r"[0-9a-f]{40}")


class PinnedConfigError(Exception):
    """pinned.toml does not pin a model. The message says which field is wrong."""


@dataclass(frozen=True)
class Pinned:
    model_id: str
    revision: str  # a full commit sha on the model's hub repo
    dtype: Dtype
    device: str
    max_new_tokens: int


def parse_pinned(text: str) -> Pinned:
    # [LAW:parse-dont-validate] a branch name or a typo'd dtype stops here, not mid-download.
    raw = tomllib.loads(text)
    model, generation = raw["model"], raw["generation"]
    if not COMMIT.fullmatch(model["revision"]):
        raise PinnedConfigError(f"model.revision must be a full commit sha, got {model['revision']!r}")
    if model["dtype"] not in get_args(Dtype):
        raise PinnedConfigError(f"model.dtype must be one of {', '.join(get_args(Dtype))}, got {model['dtype']!r}")
    max_new_tokens = generation["max_new_tokens"]
    if not (isinstance(max_new_tokens, int) and max_new_tokens > 0):
        raise PinnedConfigError(f"generation.max_new_tokens must be a positive integer, got {max_new_tokens!r}")
    return Pinned(
        model_id=model["id"],
        revision=model["revision"],
        dtype=model["dtype"],
        device=model["device"],
        max_new_tokens=max_new_tokens,
    )


def load_pinned() -> Pinned:
    return parse_pinned(files("uni").joinpath("pinned.toml").read_text())
