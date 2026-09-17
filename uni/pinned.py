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


@dataclass(frozen=True)
class Pins:
    """Every model pinned.toml pins, by name, and the name a run that names none loads."""

    default: str
    models: Mapping[str, Pinned]

    def named(self, name: str | None) -> Pinned:
        """The model pinned under `name`, or the default when `name` is None."""
        chosen = self.default if name is None else name
        if chosen not in self.models:
            raise PinnedConfigError(f"no model {chosen!r} is pinned; the models are {', '.join(self.models)}")
        return self.models[chosen]

    def writing(self, checkpoint: Mapping[str, str]) -> Pinned | None:
        """The pinned model whose weights are `checkpoint`, or None when no model pins those weights."""
        return next((pinned for pinned in self.models.values() if pinned.checkpoint == dict(checkpoint)), None)


def parse_pins(text: str) -> Pins:
    """Every pinned model, refused whole if any one of them does not pin."""
    # [LAW:parse-dont-validate] a branch-name revision, a typo'd dtype or a model nobody pinned stops
    # here, not after a download.
    raw = tomllib.loads(text)
    tables = raw.get("models")
    if not (isinstance(tables, dict) and tables and all(isinstance(table, dict) for table in tables.values())):
        raise PinnedConfigError("models must hold a table for each pinned model")
    max_new_tokens = _field(raw, ("generation", "max_new_tokens"), int)
    if max_new_tokens <= 0:
        raise PinnedConfigError(f"generation.max_new_tokens must be positive, got {max_new_tokens}")
    models = {}
    for name in tables:
        model = ("models", name)
        revision = _field(raw, (*model, "revision"), str)
        if not COMMIT.fullmatch(revision):
            raise PinnedConfigError(f"models.{name}.revision must be a full commit sha, got {revision!r}")
        models[name] = Pinned(model_id=_field(raw, (*model, "id"), str), revision=revision, dtype=_choice(raw, (*model, "dtype"), get_args(Dtype)), max_new_tokens=max_new_tokens)
    # [LAW:one-source-of-truth] what is derived from a model's weights is kept under its id, so two
    # pins of one id would share that directory, and deriving a direction under one would overwrite
    # the other's, and the file every run steered by it recorded the hash of.
    ids = [pinned.model_id for pinned in models.values()]
    twice = sorted({model_id for model_id in ids if ids.count(model_id) > 1})
    if twice:
        raise PinnedConfigError(f"{', '.join(twice)} is pinned more than once; a model's id names the directory its directions are kept in, so it is pinned once")
    default = _field(raw, ("default",), str)
    pins = Pins(default, models)
    pins.named(default)  # a default that pins nothing is refused with the rest
    return pins


def parse_pinned(text: str, name: str | None = None) -> Pinned:
    """The model pinned under `name`, or under the default name when none is given."""
    return parse_pins(text).named(name)


def load_pins() -> Pins:
    """Every model the committed pinned.toml pins."""
    return parse_pins(files("uni").joinpath("pinned.toml").read_text())


def load_pinned(name: str | None = None) -> Pinned:
    """The model pinned under `name` in the committed pinned.toml, or its default."""
    return load_pins().named(name)
