"""The steering knob: a residual-stream direction from contrastive pairs, added at the value times its length.

A contrast (uni/directions/<name>.toml) is a template, a layer, and pairs of replies to the same text,
one toward a quality and one away from it. The direction is the mean, over pairs, of the difference
between the residual stream averaged over each reply. It is derived once for each pinned model by
`uni direction <name> --model MODEL`, and kept in that model's directory beside the contrasts, as
<model id>/<name>.json, with the contrast and the checkpoint copied in: a contrast is words and holds
for any model, a direction is a vector in one model's residual stream. So a trajectory can name exactly
what steered it, and a changed checkpoint cannot be steered by a stale direction.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from uni.atomic import write_whole
from uni.maps import Turned
from uni.model import ResidualAdd
from uni.parse import ConfigError, field
from uni.pinned import Pinned, home
from uni.template import Template, TemplateError, load_templates, parse_template

if TYPE_CHECKING:
    from uni.model import Model

# A real directory, not importlib.resources: `uni direction` writes here, beside the contrast it read.
DIRECTIONS = Path(__file__).parent / "directions"


class SteerError(ConfigError):
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
    checkpoint: Mapping[str, str]  # the weights its residual stream was read from, and the only ones it steers
    vector: tuple[float, ...]  # float32 values, which JSON carries exactly

    def encode(self) -> bytes:
        contrast = self.contrast
        fields = {
            "name": contrast.name,
            "checkpoint": dict(self.checkpoint),
            "template": {"name": contrast.template.name, "text": contrast.template.text},
            "layer": contrast.layer,
            "pairs": [asdict(pair) for pair in contrast.pairs],
            "vector": list(self.vector),
        }
        return (json.dumps(fields, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.encode()).hexdigest()

    @property
    def squared_length(self) -> float:
        """What a push of 1 along this direction reads back as, along it: the unit a reading is a push in."""
        return math.fsum(value * value for value in self.vector)

    def project(self, residual: torch.Tensor) -> float:
        """How far a residual stream reaches along this direction. The direction owns its own geometry."""
        return float(self.along(residual))

    def along(self, residuals: torch.Tensor) -> torch.Tensor:
        """How far each residual stream in the last dimension of `residuals` reaches along this direction."""
        # Both sides at float32, the precision the vector is stored and committed at: a checkpoint
        # pinned at float16 would otherwise round the vector first, and the reading would carry the
        # rounding in the digits it is printed to.
        vector = torch.tensor(self.vector, device=residuals.device, dtype=torch.float32)
        return residuals.float() @ vector


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


# A contrast's name is a file name in uni/directions and nothing more: a slash or a leading dot would
# make the path it is joined into point somewhere else, and a file found there is not a contrast.
NAME = re.compile(r"[\w-][\w.-]*")


def _path(name: str, *within: str, suffix: str) -> Path:
    """The file a contrast or direction named `name` is kept in, refused unless the name is a plain one."""
    # [LAW:parse-dont-validate] [LAW:single-enforcer] every path to a contrast or direction is made here.
    if not NAME.fullmatch(name):
        raise SteerError(f"a direction is named by letters, digits, '_', '-' and '.', as a file in uni/directions is; got {name!r}")
    return DIRECTIONS.joinpath(*within, f"{name}{suffix}")


def _read(path: Path, parse: Any, fix: str) -> tuple[dict[str, Any], bytes]:
    shown = Path("uni", "directions", path.relative_to(DIRECTIONS))  # as a person finds it
    try:
        data = path.read_bytes()
        raw = parse(data.decode())
    except FileNotFoundError as error:
        known = sorted({other.stem for other in path.parent.glob("*" + path.suffix)})
        found = f"; there are {', '.join(known)}" if known else ""
        raise SteerError(f"no {path.name} in {shown.parent}{found}. {fix}") from error
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, json.JSONDecodeError) as error:
        raise SteerError(f"{shown} does not parse: {error}") from error
    if type(raw) is not dict:
        raise SteerError(f"{shown} must hold a table")
    return raw, data


def load_contrast(name: str) -> Contrast:
    raw, _ = _read(_path(name, suffix=".toml"), tomllib.loads, f"write it to describe the {name} direction")
    templates = load_templates()
    template = field(raw, "template", str, SteerError)
    if template not in templates:
        raise SteerError(f"{name}: no template {template!r}; the templates are {', '.join(templates)}")
    return _contrast(name, raw, templates[template])


def read_direction(name: str, pinned: Pinned) -> Direction:
    """The direction named `name`, refused unless it was derived on the `pinned` model."""
    raw, data = _read(_path(name, pinned.home, suffix=".json"), json.loads, f"run `uni direction {name}` with this model to derive it")
    if raw.get("checkpoint") != pinned.checkpoint:
        raise SteerError(f"{name}.json was derived on {raw.get('checkpoint')}, not the pinned checkpoint; run `uni direction {name}` with this model")
    template = field(raw, "template", dict, SteerError)
    try:
        parsed = parse_template(field(template, "name", str, SteerError), field(template, "text", str, SteerError))
    except TemplateError as error:
        raise SteerError(f"{name}: {error}") from error
    vector = field(raw, "vector", list, SteerError)
    if not (vector and all(type(value) is float and math.isfinite(value) for value in vector)):
        raise SteerError(f"{name}: vector must hold at least one value and only finite floats")
    direction = Direction(_contrast(name, raw, parsed), pinned.checkpoint, tuple(vector))
    # The contrast is copied into the file, so an edited contrast is a direction that no longer describes it.
    if direction.contrast != load_contrast(name):
        raise SteerError(f"{name}.json no longer matches the contrast in {name}.toml; run `uni direction {name}` with this model")
    # The sha256 a trajectory records is of this file, so the file has to be the one `uni direction` writes.
    if data != direction.encode():
        raise SteerError(f"{name}.json is not what `uni direction {name}` writes; re-derive it with this model rather than editing it")
    return direction


def derive(model: Model, contrast: Contrast) -> Direction:
    def reply(pair: Pair, text: str) -> torch.Tensor:
        return model.reply_residual(contrast.template.render(pair.text), text, contrast.layer)

    differences = torch.stack([reply(pair, pair.toward) - reply(pair, pair.away) for pair in contrast.pairs])
    return Direction(contrast, model.pinned.checkpoint, tuple(differences.mean(dim=0).tolist()))


def write_direction(direction: Direction) -> Path:
    # Whole or absent, as every file here is: a derivation killed mid-write leaves the committed
    # direction standing rather than replacing it with half of the next one.
    return write_whole(_path(direction.contrast.name, home(direction.checkpoint), suffix=".json"), direction.encode())


@dataclass(frozen=True)
class Steer:
    """Adds value times the direction to the residual stream leaving the direction's layer."""

    direction: Direction

    @property
    def spec(self) -> dict[str, Any]:
        # What the knob is, which turning it does not change: the value is recorded beside this,
        # once, and a sweep names the knob in its manifest before it has turned anything.
        contrast = self.direction.contrast
        return {
            "kind": "steer",
            "direction": contrast.name,
            "layer": contrast.layer,
            "sha256": self.direction.sha256,
        }

    def turn(self, value: float) -> Turned:
        layer = self.direction.contrast.layer
        return Turned(value, self.spec, (ResidualAdd(layer, value * torch.tensor(self.direction.vector)),))
