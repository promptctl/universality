"""The steering knob: a residual-stream direction from contrastive pairs, added at the value times its length.

A contrast (uni/directions/<name>.toml) is a template, a layer, and pairs of replies to the same text,
one toward a quality and one away from it. The direction is the mean, over pairs, of the difference
between the residual stream averaged over each reply. It is derived once by `uni direction <name>` and
kept beside its contrast as <name>.json, with the contrast and the pinned model copied in, so a
trajectory can name exactly what steered it and a changed model cannot be steered by a stale direction.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from uni.model import ResidualAdd
from uni.parse import field
from uni.pinned import Pinned
from uni.template import Template, TemplateError, load_templates, parse_template

if TYPE_CHECKING:
    from uni.model import Model

DIRECTIONS = Path(__file__).parent / "directions"


class SteerError(Exception):
    """A contrast or direction file is missing or malformed. The message names the file and the field."""


@dataclass(frozen=True)
class Pair:
    text: str
    toward: str
    away: str


@dataclass(frozen=True)
class Contrast:
    name: str
    template: Template
    layer: int
    pairs: tuple[Pair, ...]


@dataclass(frozen=True)
class Direction:
    contrast: Contrast
    pinned: Pinned  # the model whose residual stream it was read from, and the only one it steers
    vector: tuple[float, ...]  # float32 values, which JSON carries exactly

    def encode(self) -> bytes:
        contrast = self.contrast
        fields = {
            "name": contrast.name,
            "pinned": asdict(self.pinned),
            "template": {"name": contrast.template.name, "text": contrast.template.text},
            "layer": contrast.layer,
            "pairs": [asdict(pair) for pair in contrast.pairs],
            "vector": list(self.vector),
        }
        return (json.dumps(fields, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.encode()).hexdigest()


def _contrast(name: str, raw: Mapping[str, Any], template: Template) -> Contrast:
    # [LAW:parse-dont-validate] the contrast and its copy in a direction file are parsed by this one function.
    layer = field(raw, "layer", int, SteerError)
    if layer < 0:
        raise SteerError(f"{name}: layer must not be negative, got {layer}")
    pairs = field(raw, "pairs", list, SteerError)
    if not pairs:
        raise SteerError(f"{name}: pairs is empty; a direction needs at least one pair")
    if not all(type(pair) is dict for pair in pairs):
        raise SteerError(f"{name}: each pair must be a table of text, toward, and away")
    parsed = tuple(Pair(*(field(pair, key, str, SteerError) for key in ("text", "toward", "away"))) for pair in pairs)
    return Contrast(name, template, layer, parsed)


def _read(path: Path, parse: Any) -> dict[str, Any]:
    try:
        raw = parse(path.read_text())
    except FileNotFoundError as error:
        known = sorted({known.stem for known in DIRECTIONS.glob("*" + path.suffix)})
        raise SteerError(f"no {path.name} in uni/directions; there are {', '.join(known)}") from error
    except (tomllib.TOMLDecodeError, json.JSONDecodeError) as error:
        raise SteerError(f"uni/directions/{path.name} does not parse: {error}") from error
    if type(raw) is not dict:
        raise SteerError(f"uni/directions/{path.name} must hold a table")
    return raw


def load_contrast(name: str) -> Contrast:
    raw = _read(DIRECTIONS / f"{name}.toml", tomllib.loads)
    templates = load_templates()
    template = field(raw, "template", str, SteerError)
    if template not in templates:
        raise SteerError(f"{name}: no template {template!r}; the templates are {', '.join(templates)}")
    return _contrast(name, raw, templates[template])


def read_direction(name: str, pinned: Pinned) -> Direction:
    """The direction named `name`, refused unless it was derived on the `pinned` model."""
    raw = _read(DIRECTIONS / f"{name}.json", json.loads)
    if raw.get("pinned") != asdict(pinned):
        raise SteerError(f"{name}.json was derived on {raw.get('pinned')}, not the pinned model; run `uni direction {name}`")
    template = field(raw, "template", dict, SteerError)
    try:
        parsed = parse_template(field(template, "name", str, SteerError), field(template, "text", str, SteerError))
    except TemplateError as error:
        raise SteerError(f"{name}: {error}") from error
    vector = field(raw, "vector", list, SteerError)
    if not all(type(value) is float for value in vector):
        raise SteerError(f"{name}: vector must hold only floats")
    return Direction(_contrast(name, raw, parsed), pinned, tuple(vector))


def derive(model: Model, contrast: Contrast) -> Direction:
    def reply(pair: Pair, text: str) -> torch.Tensor:
        return model.reply_residual(contrast.template.render(pair.text), text, contrast.layer)

    differences = torch.stack([reply(pair, pair.toward) - reply(pair, pair.away) for pair in contrast.pairs])
    return Direction(contrast, model.pinned, tuple(differences.mean(dim=0).tolist()))


def write_direction(direction: Direction) -> Path:
    path = DIRECTIONS / f"{direction.contrast.name}.json"
    path.write_bytes(direction.encode())
    return path


@dataclass(frozen=True)
class Steer:
    """Adds value times the direction to the residual stream leaving the direction's layer."""

    direction: Direction

    @property
    def spec(self) -> dict[str, Any]:
        return {
            "kind": "steer",
            "direction": self.direction.contrast.name,
            "layer": self.direction.contrast.layer,
            "sha256": self.direction.sha256,
        }

    def additions(self, value: float) -> tuple[ResidualAdd, ...]:
        return (ResidualAdd(self.direction.contrast.layer, value * torch.tensor(self.direction.vector)),)
