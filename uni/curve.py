"""A response curve kept on disk: the model's answers to the pushes on a grid, at a layer, as `uni response` read them.

Reading the curve is the expensive part - a forward pass a push - and everything read off it after
that is arithmetic. So the curve is written once, committed, and read back by what fits it, and a
map fitted to it says which curve it was fitted to by the curve's own content.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uni.atomic import write_whole
from uni.parse import ConfigError, field


class CurveError(ConfigError):
    """A curve file that cannot be read as one, or does not hold the layer asked of it."""


@dataclass(frozen=True)
class Curve:
    """One layer's answers to the pushes on a grid, and the squared length that turns an answer back into a push."""

    name: str  # the file's content, hashed: two curves by one name are one curve
    layer: int
    values: tuple[float, ...]  # the pushes, rising
    readings: tuple[float, ...]
    squared_length: float


def write_curves(described: Mapping[str, Any], values: Sequence[float], curves: Mapping[int, Sequence[float]], squared_length: float, directory: Path) -> Path:
    """Write the curves read at each layer, with what they were read from, to a file named by its own bytes.

    Named by content and not by what fixed the run, as a sweep is: the readings are float32 on one
    device, and the same command on another reads a curve that differs in its last bits. Named by
    its inputs, that curve would overwrite the one a committed result was fitted to. [LAW:one-source-of-truth]
    """
    # Rising, whatever order the grid ran in: a curve is the readings as a function of the push, and
    # `read_curve` takes the ends of that order as the range a fit to it speaks for.
    order = sorted(range(len(values)), key=values.__getitem__)
    body = {"described": dict(described), "squared_length": squared_length, "values": [values[i] for i in order], "layers": {str(layer): [readings[i] for i in order] for layer, readings in curves.items()}}
    data = json.dumps(body, sort_keys=True).encode()
    return write_whole(directory / f"{hashlib.sha256(data).hexdigest()[:16]}.json", data)


def read_curve(path: Path, layer: int) -> Curve:
    """The curve at `layer` in the file at `path`, or a refusal saying what the file is not."""
    # [LAW:parse-dont-validate] a file becomes a Curve here or not at all: a fit reads nothing else.
    try:
        data = path.read_bytes()
    except OSError as error:
        raise CurveError(f"{path} cannot be read: {error.strerror}") from error
    try:
        raw = json.loads(data)
    except json.JSONDecodeError as error:
        raise CurveError(f"{path} is not JSON: {error}") from error
    if type(raw) is not dict:
        raise CurveError(f"{path} must hold a JSON object")
    values, layers = field(raw, "values", list, CurveError), field(raw, "layers", dict, CurveError)
    squared_length = field(raw, "squared_length", float, CurveError)
    if str(layer) not in layers:
        raise CurveError(f"{path} holds the curve at layer {', '.join(layers) or 'none'}, not at {layer}")
    readings = layers[str(layer)]
    if type(readings) is not list or len(readings) != len(values):
        raise CurveError(f"{path} must hold one reading at layer {layer} for each of its {len(values)} pushes")
    for name, numbers in (("push", values), ("reading", readings)):
        if not all(type(number) is float and math.isfinite(number) for number in numbers):
            raise CurveError(f"every {name} in {path} must be a finite number")
    # Rising, because a fit reads the ends as the range it says anything about, and a push it was
    # not fitted across is one it knows nothing of.
    if not values:
        raise CurveError(f"{path} holds no pushes, and a curve is read at one at least")
    if not all(before < after for before, after in zip(values, values[1:])):
        raise CurveError(f"the pushes in {path} must rise")
    if not squared_length > 0:
        raise CurveError(f"squared_length in {path} must be above zero, got {squared_length!r}")
    return Curve(hashlib.sha256(data).hexdigest()[:16], layer, tuple(values), tuple(readings), squared_length)
