"""Where a family's period-doubling cascade is, read off the values at which its top returns to itself.

PROJECT.md's third rung. A cycle through the map's critical point - the state it carries furthest,
where its slope is zero - has a multiplier of zero, the product of the slopes along it, so it is
the most stable cycle of its period there can be. Each period 2^n of a cascade has one such
superstable value, between its birth and its own doubling, and the spacings of those values
shrink by the same ratio as the doublings do: Feigenbaum's delta. Unlike a doubling, a superstable
value is found without waiting for an orbit to settle, and near a doubling an orbit settles slowly.

And the fourth rung's first number. At each superstable value the cycle's point nearest the top
is the one half a period round from it, and each doubling brings that point nearer by the same
factor, -alpha, whatever the map's shape: the cycles shrink in space as the values do along the gain.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import islice
from typing import TYPE_CHECKING

from uni.fit import Estimate, Polynomial, fit
from uni.loop import orbit
from uni.parse import ConfigError

if TYPE_CHECKING:
    from uni.loop import Map
    from uni.maps import Numbers


class CascadeError(ConfigError):
    """A grid cannot tell the superstable value of its period from one of a period dividing it."""


def returns(map: Map, numbers: Numbers, critical: str, period: int) -> dict[int, float]:
    """Where the orbit of the critical point lands after `period` steps, and after each number of steps dividing it, measured from it: F^d(x_c) - x_c.

    Zero at a superstable value of that many steps, and of opposite signs either side of it, because
    the orbit passes the top from one side and then the other. Its companion F^p(F(x_c)) - F(x_c),
    which starts a step later, would ignore a small error in x_c, since the map is flat at its top -
    but for the same reason it only touches zero, and a value it only touches cannot be bracketed.
    So the critical point is measured first (`uni critical`), and moves this by its own error.

    The steps dividing `period` come from the same orbit, on its way: a cycle through the top of d
    steps closes again after every multiple of d, so each of them is a zero this return has too.
    """
    start = numbers.read(critical)
    states = islice(orbit(map, critical), period)
    return {steps: numbers.read(state) - start for steps, state in enumerate(states, start=1) if period % steps == 0}


def superstable(values: Sequence[float], returned: Sequence[Mapping[int, float]], period: int) -> Polynomial:
    """The parabola through the return after `period` steps, whose crossing of zero is the superstable value.

    Refused when the return after fewer steps dividing `period` changes sign on the same grid: there
    the orbit of the top closes after those fewer steps, so after `period` too, and a zero found on
    the grid could be that shorter cycle's value rather than this one's. [LAW:single-enforcer]
    """
    for steps in sorted(returned[0])[:-1]:
        readings = [landed[steps] for landed in returned]
        sides = [(reading > 0) - (reading < 0) for reading in readings]
        if 0 in sides or len(set(sides)) > 1:
            raise CascadeError(
                f"the grid for period {period}, {min(values):.10g} to {max(values):.10g}, holds a superstable value of period {steps}: "
                f"the orbit of the top closes after {steps} steps there, and so after {period} as well; move the grid past it"
            )
    return fit(values, [landed[period] for landed in returned], 2)


def nearest(values: Sequence[float], returned: Sequence[Mapping[int, float]], period: int, zero: Estimate) -> Estimate:
    """F^(p/2)(x_c) - x_c at the superstable value of an even period p: how far the cycle's point nearest the top lies from it.

    Read from the returns the grid already holds, since half the period divides it, through a
    parabola evaluated at the value the return after the whole period crosses zero at. Its error is
    the parabola's own at that value and the value's error carried along the parabola's slope, added
    as if independent: both fits are through the same orbits, but the half-period's return is far
    from zero on the grid and the whole period's crosses it, so each is set by a different reading.
    """
    parabola = fit(values, [landed[period // 2] for landed in returned], 2)
    slope = parabola.at(zero.value, 1) / parabola.scale
    return Estimate(parabola.at(zero.value), math.sqrt(parabola.spread(zero.value) + (slope * zero.error) ** 2))


def quotients(values: Sequence[Estimate]) -> tuple[Estimate, ...]:
    """Each value divided by the next, with the error the two independent errors give it: the distances' ratios, which run to -alpha."""
    return tuple(Estimate(first.value / after.value, abs(first.value / after.value) * math.hypot(first.error / first.value, after.error / after.value)) for first, after in zip(values, values[1:]))


def ratios(values: Sequence[Estimate]) -> tuple[Estimate, ...]:
    """Each spacing between neighbouring values divided by the next, with the error the values' own errors give it.

    Neighbouring spacings share the value between them, so their errors are not independent: that
    value's error enters both, with opposite signs, and is counted as such.
    """
    found = []
    for first, middle, last in zip(values, values[1:], values[2:]):
        before, after = middle.value - first.value, last.value - middle.value
        ratio = before / after
        relative = (first.error**2 + middle.error**2) / before**2 + (middle.error**2 + last.error**2) / after**2 + 2 * middle.error**2 / (before * after)
        found.append(Estimate(ratio, abs(ratio) * math.sqrt(relative)))
    return tuple(found)
