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
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any

from uni.atomic import write_whole
from uni.loop import Trajectory, orbit, trajectory_name
from uni.parse import ConfigError, field

if TYPE_CHECKING:  # only to name the protocol a cell is run through; running one never needs it
    from uni.maps import Family


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
    # Finite for the reason `--value` is (cli.finite): nan and the infinities are not JSON, so a
    # manifest naming one is a file only Python reads back, and nan as an r or a knob setting
    # poisons every state after it. Refused before the ends are compared, because nan equals
    # nothing, not even itself, and would answer the comparison below with a message that
    # contradicts what the user typed.
    if not (math.isfinite(first) and math.isfinite(last)):
        raise SweepError(f"a grid runs between two finite numbers, as --value is one; got {text!r}")
    if count < 1:
        raise SweepError(f"a grid has at least one value; got {count}")
    # [LAW:no-silent-failure] the two halves of one rule: a grid names one value exactly when its
    # ends are equal. Either half alone is a typo the run would carry in silence - FROM:TO:1 is a
    # sweep the command line says is a range and the files say is a point, and FROM:FROM:200 is
    # two hundred cells that are all the same cell, run and overwritten one after another.
    if count == 1 and first != last:
        raise SweepError(f"a grid of one value is a single value; write {first}:{first}:1, not {text!r}")
    if count > 1 and first == last:
        raise SweepError(f"a grid from {first} to {last} holds one value, not {count}; write {first}:{first}:1, not {text!r}")
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

    def __post_init__(self) -> None:
        # One number, one spelling, for the reason `Trajectory` gives: a `values` of (3, 4) writes
        # "values": [3, 4], which reads back as ints the parser refuses - a manifest this sweep's
        # own reader would not take. It also makes 3 and 3.0 the one value they name one cell as,
        # rather than two spellings the distinctness check below would let through.
        object.__setattr__(self, "values", tuple(float(value) for value in self.values))
        # [LAW:parse-dont-validate] past this line a sweep's cells are distinct and every one of
        # them is a run that can be written down. A value or a start named twice is one cell named
        # twice: both runs write one file, so the second overwrites the first for no new data, the
        # count `described` prints is one the directory can never reach, and every resume runs the
        # duplicate again. It belongs to the sweep and not to `grid`, because a repeated `--start`
        # never passes through a grid at all.
        if not all(math.isfinite(value) for value in self.values):
            raise SweepError(f"a sweep's values are finite numbers, and these are not: {self.values}")
        if self.steps < 1:
            raise SweepError(f"a sweep steps each cell at least once; got {self.steps}")
        for what, given in (("values", self.values), ("starts", self.starts)):
            if not given:
                raise SweepError(f"a sweep needs at least one of {what} and this one has none")
            # Keyed by the spelling `trajectory_name` hashes, so two of these are one cell in
            # exactly the case where they are one file: 0.0 and -0.0 are equal and hash alike, and
            # would be refused as one value while naming two. [LAW:one-source-of-truth]
            seen: set[str] = set()
            for one in given:
                if repr(one) in seen:
                    raise SweepError(f"a sweep runs each cell once, and {what} names {one!r} twice")
                seen.add(repr(one))

    @property
    def name(self) -> str:
        """The directory this sweep keeps its work in, so rerunning a command resumes its own."""
        return hashlib.sha256(self.encode()).hexdigest()[:16]

    def encode(self) -> bytes:
        # Sorted keys and no timestamp: the same sweep is the same bytes, which is what makes the
        # name above stable across the runs that resume it.
        fields = {"map": self.map, "values": list(self.values), "starts": list(self.starts), "steps": self.steps}
        return (json.dumps(fields, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()

    def home(self, dir: Path) -> Path:
        """Where this sweep keeps its work, whether or not any of it has been done yet.

        Asked of the sweep so that running one and asking after one look in the same place without
        either being told where it is. [LAW:one-source-of-truth]
        """
        return dir / self.name

    @property
    def cells(self) -> Iterator[Cell]:
        """Every run this sweep is, in the order it runs them: the grid outermost, starts within."""
        for value in self.values:
            for start in self.starts:
                yield Cell(value, start, trajectory_name(self.map, value, start, self.steps))


MANIFEST = "sweep.json"


def write_sweep(sweep: Sweep, dir: Path) -> Path:
    """The sweep's own directory, with the manifest in it. Rewriting it writes the same bytes."""
    home = sweep.home(dir)
    path, written = home / MANIFEST, sweep.encode()
    # Left alone when it already says this, so resuming a finished sweep touches nothing at all.
    # The directory is named by the hash of these bytes, so a manifest here that differs was
    # edited by hand, and the command line is what a sweep is: it wins, and says nothing.
    if path.exists() and path.read_bytes() == written:
        return home
    write_whole(path, written)
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


@dataclass(frozen=True)
class Failed:
    """A cell the map could not run, and the reason it gave: all a run knows about a cell with no orbit.

    Kept as a value and never written down. A sweep directory holds trajectories and nothing else,
    so `written` below stays a question about a directory listing rather than about what is inside
    each file - and a cell that failed stays pending, which is what lets a rerun try it again once
    whatever stopped it is fixed. A failure recorded on disk would be a cell marked done by a run
    that did not do it, and nothing would ever go back for it. [LAW:one-source-of-truth]
    """

    cell: Cell
    reason: str


def run_cell(family: Family, cell: Cell, steps: int) -> Trajectory | Failed:
    """This cell's orbit, or the reason there is none, refused if it would land in the wrong file.

    A map's states can grow - a model's are its own replies - so whether step 300 still fits is
    knowable only by running to step 300, and it is the one failure the checks before the first
    cell cannot cover. So it is returned rather than raised: the cells after this one are separate
    runs of a separate map and there is nothing wrong with them, and a sweep that stopped here
    would stop at the same cell on every resume, leaving them unreachable for good.

    How far the orbit got is what tells that failure from one that is about no cell in particular,
    and the distinction is the orbit's own rather than a count of how often it has happened. A
    direction whose layer is not in the checkpoint, a vector of the wrong length: those refuse the
    first step of every cell, so nothing ever comes out, and returning them would print the one
    message a thousand times against a sweep that can never write a file. An orbit that produced
    states and then stopped outgrew something during its own run - which is exactly the failure
    this is for, and which the start cannot be checked for up front, because `holds` has already
    said the start itself fits. [LAW:types-are-the-program]

    Two refusals are never this cell's either, and are raised for the same reason. A family that
    cannot make a map at a value it was offered before the run began cannot make one at any of
    them; and a map that describes itself differently than it did when the sweep named its cells
    writes files no run of this sweep will ever look for.
    """
    # Outside the try below, because a family refusing a value it already accepted is not this
    # cell failing - it is the check before the first cell and the map disagreeing about the grid.
    map = family.at(cell.value)
    states: list[str] = []
    try:
        # Collected as they land rather than in one `tuple(...)`, so that when the map stops the
        # run still holds what came out of it, which is what decides the answer below.
        for state in islice(orbit(map, cell.start), steps):
            states.append(state)
    except ConfigError as error:
        if not states:
            raise
        return Failed(cell, str(error))
    # The map's own description and its own value, the way `uni loop` records them, so what the
    # file says the orbit ran at is what it ran at. [LAW:one-source-of-truth]
    trajectory = Trajectory(map.spec, map.value, cell.start, tuple(states))
    # [LAW:no-silent-failure] the sweep named this cell's file before running it, and resumption
    # is that name coming true. A map that came back set to something other than what it was asked
    # for writes a file this sweep cannot find - and then every rerun runs the cell again, for
    # ever, against a total that never lands.
    if trajectory.name != cell.name:
        raise SweepError(
            f"the cell at {cell.value} writes {trajectory.name}, not the {cell.name} this sweep is looking for: "
            "the map does not describe itself the same way twice"
        )
    return trajectory


def written(cell: Cell, home: Path) -> bool:
    """Whether this cell's orbit is on disk, which is the whole of what done means here.

    A file is there or it is not, and `write_whole` renames a finished file over its name, so a run
    killed part way through leaves no half-written cell to mistake for a done one. One predicate,
    because a sweep runs what is missing and a plot draws what is there, and those two must not be
    able to disagree about which cells those are. [LAW:one-source-of-truth]
    """
    return (home / cell.name).exists()


def pending(sweep: Sweep, home: Path) -> tuple[Cell, ...]:
    """The cells with no trajectory on disk yet: what running this sweep would do."""
    return tuple(cell for cell in sweep.cells if not written(cell, home))


def finished(sweep: Sweep, home: Path) -> tuple[Cell, ...]:
    """The cells whose trajectory is on disk: what a picture of this sweep can be drawn from.

    A sweep is normally incomplete while it runs, and a picture of what is there so far is worth
    having, so this is the answer rather than a refusal. It is in sweep order, so a figure drawn
    twice from the same directory puts its points down in the same order both times.
    """
    return tuple(cell for cell in sweep.cells if written(cell, home))


def described(sweep: Sweep, cells: Sequence[Cell]) -> str:
    """How much of the sweep is left, in one line."""
    total = len(sweep.values) * len(sweep.starts)
    return f"{total - len(cells)} of {total} cells done, {len(cells)} to run"
