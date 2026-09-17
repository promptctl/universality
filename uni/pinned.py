"""The pinned model configurations, parsed from pinned.toml: every model a run may name, and the one it runs without naming one."""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Literal, get_args

from uni.parse import ConfigError

Dtype = Literal["float32", "float16", "bfloat16"]

COMMIT = re.compile(r"[0-9a-f]{40}")


class PinnedConfigError(ConfigError):
    """pinned.toml does not pin the model asked for. The message says which field is wrong."""


@dataclass(frozen=True)
class Pinned:
    model_id: str
    revision: str  # a full commit sha on the model's hub repo
    dtype: Dtype
    max_new_tokens: int

    @property
    def checkpoint(self) -> dict[str, str]:
        """What fixes the weights and their arithmetic, and so fixes any direction read from them."""
        return {"model_id": self.model_id, "revision": self.revision, "dtype": self.dtype}

    @property
    def home(self) -> str:
        return home(self.checkpoint)


def home(checkpoint: Mapping[str, str]) -> str:
    """The directory name what is derived from a checkpoint's weights is kept under: its model's id, with the slash made a file name's.

    [LAW:one-source-of-truth] read off the checkpoint and not off the name pinned.toml gives it, so a
    file kept there says which weights it belongs to by where it is as well as by what it records,
    and a direction read back is looked for where the weights it records put it.
    """
    return checkpoint["model_id"].replace("/", "--")


def _field(raw: dict[str, Any], path: tuple[str, ...], kind: type) -> Any:
    """The value at `path`, a key for each table on the way down: keys and not a dotted string, since a model's name may hold dots."""
    *tables, key = path
    section: Any = raw
    for table in tables:
        section = section.get(table) if isinstance(section, dict) else None
    if not (isinstance(section, dict) and key in section):
        raise PinnedConfigError(f"{'.'.join(path)} is missing")
    value = section[key]
    # Exact type, so a TOML boolean is not taken for an integer.
    if type(value) is not kind:
        raise PinnedConfigError(f"{'.'.join(path)} must be a {kind.__name__}, got {value!r}")
    return value


def _choice(raw: dict[str, Any], path: tuple[str, ...], choices: tuple[str, ...]) -> Any:
    value = _field(raw, path, str)
    if value not in choices:
        raise PinnedConfigError(f"{'.'.join(path)} must be one of {', '.join(choices)}, got {value!r}")
    return value


def parse_pinned(text: str, name: str | None = None) -> Pinned:
    """The model pinned under `name`, or under the default name when none is given."""
    # [LAW:parse-dont-validate] a branch-name revision, a typo'd dtype or a model nobody pinned stops
    # here, not after a download.
    raw = tomllib.loads(text)
    models = raw.get("models")
    if not (isinstance(models, dict) and models and all(isinstance(table, dict) for table in models.values())):
        raise PinnedConfigError("models must hold a table for each pinned model")
    chosen = _field(raw, ("default",), str) if name is None else name
    if chosen not in models:
        raise PinnedConfigError(f"no model {chosen!r} is pinned; the models are {', '.join(models)}")
    model = ("models", chosen)
    revision = _field(raw, (*model, "revision"), str)
    if not COMMIT.fullmatch(revision):
        raise PinnedConfigError(f"models.{chosen}.revision must be a full commit sha, got {revision!r}")
    max_new_tokens = _field(raw, ("generation", "max_new_tokens"), int)
    if max_new_tokens <= 0:
        raise PinnedConfigError(f"generation.max_new_tokens must be positive, got {max_new_tokens}")
    return Pinned(
        model_id=_field(raw, (*model, "id"), str),
        revision=revision,
        dtype=_choice(raw, (*model, "dtype"), get_args(Dtype)),
        max_new_tokens=max_new_tokens,
    )


def load_pinned(name: str | None = None) -> Pinned:
    """The model pinned under `name` in the committed pinned.toml, or its default."""
    return parse_pinned(files("uni").joinpath("pinned.toml").read_text(), name)
