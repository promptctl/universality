"""A sweep: one map run at every value on a grid, from every start, kept under one directory.

Resumable, and by construction rather than by bookkeeping. A trajectory's file name is a hash of
what fixes the orbit, and a sweep knows all of that before it runs a cell, so "has this cell been
done" is "is that file there" - a question asked of the work itself. The manifest records what the
sweep is and never what it has finished: a record of progress kept beside the files it describes
is a second clock, and a run killed between writing a trajectory and updating it would leave the
two disagreeing about work that is plainly there. [LAW:one-source-of-truth]
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uni.loop import trajectory_name
from uni.parse import ConfigError, field


class SweepError(ConfigError):
    """A sweep cannot be described as asked, or a directory does not hold one."""


@dataclass(frozen=True)
class Cell:
    """One run of the sweep: the map's parameter, the start, and the file that orbit writes."""

    value: float
    start: str
    name: str


def grid(text: str) -> tuple[float, ...]:
    """The values `FROM:TO:COUNT` names, `TO` included.

    Written out rather than recomputed later: the manifest carries the numbers themselves, so
    what a resumed run looks for is what the first run wrote, not an arithmetic that has to land
    on the same floats twice.
    """
    parts = text.split(":")
    if len(parts) != 3:
        raise SweepError(f"a grid is FROM:TO:COUNT, as in 2.8:4.0:200; got {text!r}")
    try:
        first, last, count = float(parts[0]), float(parts[1]), int(parts[2])
    except ValueError as error:
        raise SweepError(f"a grid is FROM:TO:COUNT, two numbers and a whole count; got {text!r}") from error
    if count < 1:
        raise SweepError(f"a grid has at least one value; got {count}")
    # [LAW:no-silent-failure] one value cannot also be a range. Taking FROM and dropping TO would
    # run a sweep the command line says is a hundred wide and the files say is one.
    if count == 1 and first != last:
        raise SweepError(f"a grid of one value is a single value; write {first}:{first}:1, not {text!r}")
    if count == 1:
        return (first,)
    step = (last - first) / (count - 1)

    def point(index: int) -> float:
        # The ends are the numbers that were asked for, exactly, and not what the arithmetic
        # lands near: first + step * (count - 1) misses `last` for about one grid in twenty, and
        # a logistic sweep to r = 4 - the edge of the map's range, and the sweep worth running -
        # would then end on an r the map refuses, killing the run at its last cell.
        if index == 0:
            return first
        if index == count - 1:
            return last
        return first + step * index

    return tuple(point(index) for index in range(count))


@dataclass(frozen=True)
class Sweep:
    """Everything that fixes a sweep: the map, the values, the starts, and how long each orbit runs."""

    map: Mapping[str, Any]  # the map's description, which a value does not change
    values: tuple[float, ...]
    starts: tuple[str, ...]
    steps: int

    @property
    def name(self) -> str:
        """The directory this sweep keeps its work in, so rerunning a command resumes its own."""
        return hashlib.sha256(self.encode()).hexdigest()[:16]

    def encode(self) -> bytes:
        # Sorted keys and no timestamp: the same sweep is the same bytes, which is what makes the
        # name above stable across the runs that resume it.
        fields = {"map": self.map, "values": list(self.values), "starts": list(self.starts), "steps": self.steps}
        return (json.dumps(fields, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()

    @property
    def cells(self) -> Iterator[Cell]:
        """Every run this sweep is, in the order it runs them: the grid outermost, starts within."""
        for value in self.values:
            for start in self.starts:
                yield Cell(value, start, trajectory_name(self.map, value, start, self.steps))


MANIFEST = "sweep.json"


def write_sweep(sweep: Sweep, dir: Path) -> Path:
    """The sweep's own directory, with the manifest in it. Rewriting it writes the same bytes."""
    home = dir / sweep.name
    home.mkdir(parents=True, exist_ok=True)
    path, written = home / MANIFEST, sweep.encode()
    # Left alone when it already says this, so resuming a finished sweep touches nothing at all.
    # The directory is named by the hash of these bytes, so a manifest here that differs was
    # edited by hand, and the command line is what a sweep is: it wins, and says nothing.
    if path.exists() and path.read_bytes() == written:
        return home
    partial = home / (MANIFEST + ".partial")
    partial.write_bytes(written)
    partial.replace(path)
    return home


def read_sweep(path: Path) -> Sweep:
    """The sweep a manifest describes, refused unless the file holds one."""
    try:
        raw = json.loads(path.read_bytes())
    except OSError as error:  # the path is one the user typed, so a mistyped one is theirs to fix
        raise SweepError(f"{path} cannot be read: {error.strerror}") from error
    except json.JSONDecodeError as error:
        raise SweepError(f"{path} is not JSON: {error}") from error
    if not isinstance(raw, dict):
        raise SweepError(f"{path} must hold a JSON object, not {type(raw).__name__}")
    values = field(raw, "values", list, SweepError)
    starts = field(raw, "starts", list, SweepError)
    if not all(type(value) is float for value in values):
        raise SweepError("values must all be numbers")
    if not all(type(start) is str for start in starts):
        raise SweepError("starts must all be strings")
    return Sweep(
        field(raw, "map", dict, SweepError),
        tuple(values),
        tuple(starts),
        field(raw, "steps", int, SweepError),
    )


def pending(sweep: Sweep, home: Path) -> tuple[Cell, ...]:
    """The cells with no trajectory on disk yet.

    A file is there or it is not, and `write_trajectory` renames a finished file over its name, so
    a run killed part way through leaves no half-written cell to mistake for a done one.
    """
    return tuple(cell for cell in sweep.cells if not (home / cell.name).exists())


def described(sweep: Sweep, cells: Sequence[Cell]) -> str:
    """How much of the sweep is left, in one line."""
    total = len(sweep.values) * len(sweep.starts)
    return f"{total - len(cells)} of {total} cells done, {len(cells)} to run"
