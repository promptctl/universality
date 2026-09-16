"""The numbers a sweep's two pictures are drawn from, read off the trajectories on disk.

Both pictures are the same kind of thing - points with an x, a y, and something that colours them
- so nothing here knows how to draw, and nothing that draws knows what a sweep is. A picture is
therefore a value this module returns rather than a file it writes. [LAW:effects-at-boundaries]

Neither picture asks which map ran. The return map is an observable's sequence plotted against
itself one step later, and the orbit diagram is the same readings against the value they were
taken at; both are true of a number read off any orbit, so a model sweep and a logistic one go
down the same path. [LAW:composability]
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from uni.loop import read_trajectory
from uni.observe import Observable, ObserveError, Weights, observables, readings, steps
from uni.sweep import Sweep, finished


@dataclass(frozen=True)
class Point:
    """One dot: where it goes, and the number that decides its colour.

    The shade is a number and not a colour, so this module never says what anything looks like.
    Every point of a picture carrying the same shade is how a picture asks for one colour - there
    is no second kind of picture to branch on. [LAW:dataflow-not-control-flow]
    """

    x: float
    y: float
    shade: float


@dataclass(frozen=True)
class Picture:
    """Points and what they mean: enough to draw, and nothing about how."""

    points: tuple[Point, ...]
    title: str
    x_label: str
    y_label: str
    shade_label: str | None  # None when every point is the one shade, so there is nothing to key


@dataclass(frozen=True)
class Readings:
    """One cell's orbit, read for one observable: the value it ran at, and the numbers in order."""

    value: float  # the map's parameter - r, or the knob's setting - which is the orbit diagram's x
    numbers: tuple[float, ...]


def named(available: Sequence[Observable], name: str) -> Observable:
    """The observable that goes by that name, refused unless this orbit answers for it."""
    # [LAW:parse-dont-validate] the name becomes an observable here or not at all, so nothing
    # downstream holds a string that may or may not be one.
    for observable in available:
        if observable.name == name:
            return observable
    raise ObserveError(f"no observable {name!r} for this sweep; it reads {', '.join(o.name for o in available)}")


def read(sweep: Sweep, home: Path, name: str, burn_in: int) -> tuple[Readings, ...]:
    """Every finished cell's readings for one observable, with the transient dropped.

    The burn-in is dropped here rather than by each picture, because it means one thing - the
    steps before the orbit settled are not what either picture is about - and a picture that
    dropped its own would be a second place deciding what settled means. [LAW:single-enforcer]

    It means the same thing it means to `uni observe`, which is why it is written as a test on the
    step's own number rather than as a slice: `detect` passes over a sequence whose element 0 is
    the start, so its `--burn-in N` keeps step N onward. A slice of `steps()` starts counting at
    step 1 and would keep step N + 1, so the period a person reads off one command would be
    measured over different states than the picture drawn by the other. [LAW:one-source-of-truth]
    """
    # One checkpoint for the whole picture, not one per cell: what every cell of a sweep is read
    # through is the same model, because the sweep's own description says so. [LAW:carrying-cost]
    weights = Weights()
    out = []
    for cell in finished(sweep, home):
        trajectory = read_trajectory(home / cell.name)
        observable = named(observables(trajectory, weights), name)
        settled = [step for step in steps(trajectory) if step.index >= burn_in]
        # [LAW:no-silent-failure] `readings` names the step it could not read, which was the whole
        # story when the caller was one named trajectory. Over a sweep it is the cell that says
        # which file to go and look at.
        try:
            numbers = tuple(readings((observable,), step)[0] for step in settled)
        except ObserveError as error:
            raise ObserveError(f"{cell.name}: {error}") from error
        out.append(Readings(trajectory.value, numbers))
    return tuple(out)


def return_map(series: Sequence[Readings], name: str) -> Picture:
    """Each reading against the one after it: the map itself, drawn.

    Coloured by the value the orbit ran at, so a sweep of one value is one colour and a grid is a
    fan of them - the same picture the ticket asks for "at one knob value or across the grid",
    with nothing to choose between.
    """
    points = tuple(
        Point(before, after, one.value)
        for one in series
        for before, after in zip(one.numbers, one.numbers[1:])
    )
    return Picture(points, f"return map: {name}", f"{name} at step n", f"{name} at step n + 1", "value")


def orbit_diagram(series: Sequence[Readings], name: str) -> Picture:
    """Every settled reading against the value it was taken at: the bifurcation diagram.

    One shade for every point, because the value is already the x axis and colouring by it a
    second time would say nothing twice. This is the picture PROJECT.md's Rung 1 is about:
    period doubling is branches splitting as the value grows.
    """
    points = tuple(Point(one.value, number, 0.0) for one in series for number in one.numbers)
    return Picture(points, f"orbit diagram: {name}", "value", name, None)
