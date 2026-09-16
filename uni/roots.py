"""Bisection: where a function of one number changes sign, found to the resolution its arguments are spelled at."""

from __future__ import annotations

from collections.abc import Callable


def bisect(function: Callable[[float], float], low: float, at_low: float, high: float, at_high: float, spell: Callable[[float], float] = float) -> float:
    """The number between `low` and `high` the function is least far from zero at, halving until the midpoint spells an end.

    The ends come with their readings, which the caller has already taken to decide it holds a
    bracket: they must lie strictly on opposite sides of zero, and how to refuse ends that do not
    is the caller's to say, in its own terms. `spell` is how a midpoint is written before it is
    read - to four decimals for a map that writes its states so, exactly for a float - and the
    search ends where that spelling can no longer tell a midpoint from an end.
    """
    while True:
        middle = spell((low + high) / 2)
        if middle in (low, high):
            return low if abs(at_low) <= abs(at_high) else high
        at_middle = function(middle)
        if at_middle == 0:
            return middle
        if (at_middle > 0) == (at_low > 0):
            low, at_low = middle, at_middle
        else:
            high, at_high = middle, at_middle
