"""Least-squares polynomials through measured readings, and where one or its slope crosses zero, with an error.

A measured map's readings are not exact, so a number read off them is worth only as much as its
error says. Both numbers the cascade rests on are crossings of a fitted polynomial: a superstable
value is where a parabola through the return of the map's top crosses zero, and the top itself is
where the slope of a cubic through the map crosses zero. The error of each comes from the scatter
of the readings about the fit, carried through the fit's coefficients to where the crossing lies.

The scatter measures what is random. A curve the fit is one degree too low to follow is not, and
moves the crossing by more than the error says: through the logistic map's exact returns, a line
misplaces every superstable value by three times the error it reports, and a parabola by one. So
each fit here is a degree above the shape it is looking for - a crossing, a turn.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from uni.parse import ConfigError
from uni.roots import bisect


class FitError(ConfigError):
    """The readings cannot be fitted, or the fit does not cross zero once on its grid. The message says which."""


@dataclass(frozen=True)
class Estimate:
    """A measured number and its standard error."""

    value: float
    error: float


@dataclass(frozen=True)
class Polynomial:
    """c_0 + c_1 u + c_2 u^2 + ... in u = (x - center) / scale, fitted to readings.

    In u rather than x, with u running from -1 to 1 across the grid, so the equations the fit solves
    stay well conditioned for a grid far from zero and narrow, which is the only kind a crossing is
    looked for on.
    """

    center: float
    scale: float
    coefficients: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]  # of the coefficients, from the scatter
    low: float
    high: float
    scatter: float  # rms of the readings about the fit, in the readings' own units

    def terms(self, x: float, derivative: int) -> tuple[float, ...]:
        """What each coefficient contributes, per unit of itself, to the fit's `derivative`-th derivative in u at x."""
        u = (x - self.center) / self.scale
        return tuple(math.perm(power, derivative) * u ** (power - derivative) if power >= derivative else 0.0 for power in range(len(self.coefficients)))

    def at(self, x: float, derivative: int = 0) -> float:
        return math.fsum(coefficient * term for coefficient, term in zip(self.coefficients, self.terms(x, derivative)))


def fit(values: Sequence[float], readings: Sequence[float], degree: int) -> Polynomial:
    """The least-squares polynomial of `degree` through the readings taken at `values`."""
    size = degree + 1
    # [LAW:no-silent-failure] a fit with no readings to spare passes through every one of them, so
    # its scatter is zero and every error it reports is zero, whatever the readings were.
    if len(values) <= size:
        raise FitError(f"a polynomial of degree {degree} fitted to {len(values)} readings has none left over to measure its scatter by; give at least {size + 1}")
    low, high = min(values), max(values)
    center, scale = (low + high) / 2, (high - low) / 2
    rows = [[((value - center) / scale) ** power for power in range(size)] for value in values] if scale else []
    normal = [[math.fsum(row[i] * row[j] for row in rows) for j in range(size)] for i in range(size)]
    inverse = invert(normal, values)
    projected = [math.fsum(row[i] * reading for row, reading in zip(rows, readings)) for i in range(size)]
    coefficients = tuple(math.fsum(inverse[i][j] * projected[j] for j in range(size)) for i in range(size))
    residual = math.fsum((reading - math.fsum(c * t for c, t in zip(coefficients, row))) ** 2 for row, reading in zip(rows, readings))
    variance = residual / (len(values) - size)
    covariance = tuple(tuple(variance * entry for entry in row) for row in inverse)
    return Polynomial(center, scale, coefficients, covariance, low, high, math.sqrt(residual / len(values)))


def invert(matrix: list[list[float]], values: Sequence[float]) -> list[list[float]]:
    """The inverse of the fit's normal equations, by Gauss-Jordan elimination with partial pivoting."""
    size = len(matrix)
    work = [row[:] + [float(i == j) for j in range(size)] for i, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(work[row][column]))
        if work[pivot][column] == 0:
            distinct = len(set(values))
            raise FitError(f"a polynomial of degree {size - 1} needs readings at {size} distinct values at least, and these are at {distinct}")
        work[column], work[pivot] = work[pivot], work[column]
        lead = work[column][column]
        work[column] = [entry / lead for entry in work[column]]
        for row in range(size):
            if row != column and work[row][column]:
                factor = work[row][column]
                work[row] = [entry - factor * leading for entry, leading in zip(work[row], work[column])]
    return [row[size:] for row in work]


# The two derivatives a crossing is looked for in, by what a crossing of each is: where a curve
# is zero, and where it is highest or lowest.
NAMES = {0: "fitted curve", 1: "slope of the fitted curve"}


def crossing(polynomial: Polynomial, derivative: Literal[0, 1]) -> Estimate:
    """Where the fit, or its slope, crosses zero on its grid, refused unless it does so exactly once.

    Looked for on 64 even steps across the grid rather than at the readings' own values, so a fit
    through few readings is searched as finely as one through many. Two crossings closer together
    than a step leave no change of sign between them, and are refused as none rather than taken as
    one. The error is the fit's own: how far the scatter moves the coefficients, carried to the
    crossing by how steeply it is crossed. A gentle crossing is placed loosely, and says so.
    """
    points = [polynomial.low + (polynomial.high - polynomial.low) * index / 64 for index in range(65)]
    readings = [polynomial.at(point, derivative) for point in points]
    found = []
    for index, (point, reading) in enumerate(zip(points, readings)):
        if reading == 0:
            found.append(point)
        elif index and reading * readings[index - 1] < 0:
            found.append(bisect(lambda x: polynomial.at(x, derivative), points[index - 1], readings[index - 1], point, reading))
    if len(found) != 1:
        raise FitError(f"the {NAMES[derivative]} crosses zero {len(found)} times between {polynomial.low:.8g} and {polynomial.high:.8g}; move the grid so it crosses once")
    (where,) = found
    steepness = polynomial.at(where, derivative + 1)
    if steepness == 0:
        raise FitError(f"the {NAMES[derivative]} touches zero at {where:.8g} without passing through it, so its crossing has no error to give")
    terms = polynomial.terms(where, derivative)
    spread = math.fsum(terms[i] * polynomial.covariance[i][j] * terms[j] for i in range(len(terms)) for j in range(len(terms)))
    # A variance, so never below zero; fsum of a form that is exactly zero can still land a hair under it.
    return Estimate(where, polynomial.scale * math.sqrt(max(spread, 0.0)) / abs(steepness))
