"""A response curve kept on disk: the model's answers to the pushes on a grid, at a layer, as `uni response` read them.

Reading the curve is the expensive part - a forward pass a push - and everything read off it after
that is arithmetic. So the curve is written once, committed, and read back by what fits it, and a
map fitted to it says which curve it was fitted to by the curve's own content.
"""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from uni.atomic import write_whole
from uni.parse import ConfigError
from uni.parse import field as required


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
    described: Mapping[str, Any] = field(default_factory=dict)  # what the readings were read from, as the file records it

    @property
    def reading(self) -> str | None:
        """What each reading is, as `uni temperature` records it: a mean or a spread. `uni response` records none."""
        reading = self.described.get("reading")
        return reading if type(reading) is str else None

    @property
    def source(self) -> Mapping[str, Any]:
        """What was pushed and read, whatever was taken of the draw: two curves of one source are answers and noise of one loop."""
        return {key: value for key, value in self.described.items() if key not in ("reading", "temperature")}

    def at(self, push: float) -> float:
        """The reading at `push`, on the straight line between the readings either side of it, refused outside the pushes read."""
        # [LAW:no-silent-failure] a curve says nothing past the pushes it was read at, as a series fitted to it does not.
        if not self.values[0] <= push <= self.values[-1]:
            raise CurveError(f"a push of {push!r} lies outside curve {self.name}'s pushes, {self.values[0]:g} to {self.values[-1]:g}")
        right = min(bisect_right(self.values, push), len(self.values) - 1)  # the first push past this one, or the last
        if right == 0:  # a curve of one push, read at it
            return self.readings[0]
        before, after = self.values[right - 1], self.values[right]
        share = (push - before) / (after - before)
        return self.readings[right - 1] + share * (self.readings[right] - self.readings[right - 1])


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
    values, layers = required(raw, "values", list, CurveError), required(raw, "layers", dict, CurveError)
    squared_length = required(raw, "squared_length", float, CurveError)
    described = raw.get("described", {})
    if type(described) is not dict:
        raise CurveError(f"described in {path} must be a JSON object")
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
    return Curve(hashlib.sha256(data).hexdigest()[:16], layer, tuple(values), tuple(readings), squared_length, described)


def read_answers(path: Path, layer: int) -> Curve:
    """The curve at `layer` in the file at `path`, refused if its readings are spreads: what a map is fitted to.

    [LAW:no-silent-failure] `uni temperature` writes a curve of spreads beside each curve of means,
    and a series fitted to the one reads as a map as readily as one fitted to the other.
    """
    curve = read_curve(path, layer)
    if curve.reading == "spread":
        raise CurveError(f"{path} holds spreads, and a map is fitted to answers: a curve `uni response` wrote, or the means `uni temperature` wrote beside it")
    return curve


def read_spreads(path: Path, layer: int, answers: Curve) -> Curve:
    """The curve at `layer` in the file at `path`, refused unless its readings are spreads of the loop `answers` was read from.

    [LAW:single-enforcer] the one place noise is read from a file. A curve of means or of answers
    taken for spreads would be noise the size of the answer itself, and spreads read along another
    direction or prompt noise of some other loop's size, each carried and printed as if measured.
    [LAW:no-silent-failure]
    """
    curve = read_curve(path, layer)
    if curve.reading != "spread":
        held = "readings it does not name" if curve.reading is None else f"{curve.reading}s"
        raise CurveError(f"{path} holds {held}, and noise is read from the spreads `uni temperature` writes")
    if curve.source != answers.source or curve.squared_length != answers.squared_length:
        raise CurveError(f"{path} holds the spreads of another loop than curve {answers.name} was read from: the prompt, knob, model or squared length they record differ")
    return curve
