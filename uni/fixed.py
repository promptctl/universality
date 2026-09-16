"""Where a map whose states are numbers holds still, and how steeply it moves there.

PROJECT.md's second rung, read off the map itself rather than off an orbit. An orbit near the flip
settles slowly, and on a map whose states are written to a few decimals it can alternate between
two neighbouring spellings of one push, which a period detector counts as a period 2 the map does
not have. The slope at the fixed point has neither trouble: it is -1 exactly where the fixed point
gives way to a period-2 orbit, whatever the orbit is doing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from uni.parse import ConfigError

if TYPE_CHECKING:
    from uni.loop import Map
    from uni.maps import Numbers


class FixedError(ConfigError):
    """No fixed point is bracketed, or the slope is asked for at a step finer than the map writes."""


def moved(map: Map, numbers: Numbers, x: float) -> float:
    """How far one step carries the state that spells `x`: F(x) - x."""
    return numbers.read(map.step(numbers.write(x))) - x


def fixed_point(map: Map, numbers: Numbers, low: float, high: float) -> float:
    """The number between `low` and `high` the map carries least far, found by bisection on the sign of F(x) - x.

    Bisected on states rather than on floats: each midpoint is the state the map would write for
    it, and the search ends when the midpoint spells the same state as an end, so it ends at the
    map's own resolution - float64's for the logistic map, four decimals for the response map.
    """
    below, above = (numbers.read(numbers.write(end)) for end in (low, high))
    at_below, at_above = moved(map, numbers, below), moved(map, numbers, above)
    # [LAW:no-silent-failure] a bracket is two states carried strictly in opposite directions. Ends
    # carried the same way would still converge - on an end, reported as a fixed point the map
    # does not have there - and an end the map already holds still would be returned as the fixed
    # point asked for between them, whichever one the bracket was drawn around.
    if not min(at_below, at_above) < 0 < max(at_below, at_above):
        raise FixedError(
            f"at {map.value:g} the map carries {below:g} by {at_below:+.3g} and {above:g} by {at_above:+.3g}; "
            "a bracket is two states it carries in opposite directions, so no fixed point is bracketed between them"
        )
    while True:
        middle = numbers.read(numbers.write((below + above) / 2))
        if middle in (below, above):
            return below if abs(at_below) <= abs(at_above) else above
        at_middle = moved(map, numbers, middle)
        if at_middle == 0:
            return middle
        if (at_middle > 0) == (at_below > 0):
            below, at_below = middle, at_middle
        else:
            above, at_above = middle, at_middle


def slope(map: Map, numbers: Numbers, x: float, step: float) -> float:
    """F'(x) as the central difference across the states the map writes for x - step and x + step."""
    below, above = (numbers.read(numbers.write(x + offset)) for offset in (-step, step))
    if below == above:
        raise FixedError(f"a step of {step:g} is finer than the map writes its states, so both sides of {x:g} are one state")
    rise = moved(map, numbers, above) + above - (moved(map, numbers, below) + below)
    return rise / (above - below)


def crossings(values: Sequence[float], readings: Sequence[float], level: float) -> tuple[float, ...]:
    """Where the readings pass from one side of `level` to the other.

    Between two neighbouring readings either side of it, the place is a straight line between their
    values. A run of readings exactly on the level is placed at its first value, and passes through
    only if what borders it differs: a touch from one side and back is not a crossing, and a run at
    either end of the grid, whose other border is unseen, is one - unless every reading is on it.
    """
    sides = [(reading > level) - (reading < level) for reading in readings]
    found = []
    for index, side in enumerate(sides):
        before = sides[index - 1] if index else 0
        if side == 0 and (index == 0 or before != 0):
            end = next((later for later in range(index, len(sides)) if sides[later] != 0), len(sides))
            if before != (sides[end] if end < len(sides) else 0):
                found.append(values[index])
        elif side * before < 0:
            v0, v1, r0, r1 = values[index - 1], values[index], readings[index - 1], readings[index]
            found.append(v0 + (level - r0) * (v1 - v0) / (r1 - r0))
    return tuple(found)
