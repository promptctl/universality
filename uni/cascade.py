"""Where a family's period-doubling cascade is, read off the values at which its top returns to itself.

PROJECT.md's third rung. A cycle through the map's critical point - the state it carries furthest,
where its slope is zero - has a multiplier of zero, the product of the slopes along it, so it is
the most stable cycle of its period there can be. Each period 2^n of a cascade has one such
superstable value, between its birth and its own doubling, and the spacings of those values
shrink by the same ratio as the doublings do: Feigenbaum's delta. Unlike a doubling, a superstable
value is found without waiting for an orbit to settle, and near a doubling an orbit settles slowly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import islice
from typing import TYPE_CHECKING

from uni.fit import Estimate
from uni.loop import orbit

if TYPE_CHECKING:
    from uni.loop import Map
    from uni.maps import Numbers


def returned(map: Map, numbers: Numbers, critical: str, period: int) -> float:
    """Where `period` steps from the critical point land, measured from it: F^p(x_c) - x_c.

    Zero at a superstable value, and of opposite signs either side of it, because the orbit passes
    the top from one side and then the other. Its companion F^p(F(x_c)) - F(x_c), which starts a
    step later, would ignore a small error in x_c, since the map is flat at its top - but for the
    same reason it only touches zero, and a value it only touches cannot be bracketed.
    So the critical point is measured first (`uni critical`), and moves this by its own error.
    """
    (after,) = islice(orbit(map, critical), period - 1, period)
    return numbers.read(after) - numbers.read(critical)


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
