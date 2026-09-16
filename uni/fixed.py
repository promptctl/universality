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
    # [LAW:no-silent-failure] with no change of sign between the ends, bisection would still
    # converge - on an end, reported as a fixed point the map does not have there.
    if (at_below > 0) == (at_above > 0) and at_below and at_above:
        raise FixedError(f"at {map.value:g} the map carries both {below:g} and {above:g} the same way, so no fixed point is bracketed between them")
    while at_below and at_above:
        middle = numbers.read(numbers.write((below + above) / 2))
        if middle in (below, above):
            break
        at_middle = moved(map, numbers, middle)
        if (at_middle > 0) == (at_below > 0):
            below, at_below = middle, at_middle
        else:
            above, at_above = middle, at_middle
    return below if abs(at_below) <= abs(at_above) else above


def slope(map: Map, numbers: Numbers, x: float, step: float) -> float:
    """F'(x) as the central difference across the states the map writes for x - step and x + step."""
    below, above = (numbers.read(numbers.write(x + offset)) for offset in (-step, step))
    if below == above:
        raise FixedError(f"a step of {step:g} is finer than the map writes its states, so both sides of {x:g} are one state")
    rise = moved(map, numbers, above) + above - (moved(map, numbers, below) + below)
    return rise / (above - below)


def crossings(values: Sequence[float], readings: Sequence[float], level: float) -> tuple[float, ...]:
    """Where the readings pass through `level`, each placed by a straight line between the two values either side."""
    found = []
    for (v0, r0), (v1, r1) in zip(zip(values, readings), zip(values[1:], readings[1:])):
        if (r0 - level) * (r1 - level) < 0 or (r1 == level and r0 != level):
            found.append(v0 + (level - r0) * (v1 - v0) / (r1 - r0))
    return tuple(found)
