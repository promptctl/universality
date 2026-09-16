"""A smooth function through a response curve's readings: the model's answer with its float32 roughness fitted away.

Read directly, the response map's cascade stops at period 64: its returns scatter 8e-4 there,
the size of what distinguishes one superstable gain from the next. A Chebyshev series fitted to
the sampled curve, to a residual no larger than the readings' own jitter, is a map with the same
shape and no roughness, whose cascade can be followed as far as float64 reaches. What it says
about the model rests on how well it fits, so the fit is reported beside it, and the cascade is
read at more than one degree to show which ratios depend on the fit's fidelity and which do not.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

from uni.curve import Curve
from uni.parse import ConfigError


class SmoothError(ConfigError):
    """The curve cannot be fitted at the degree asked."""


@dataclass(frozen=True)
class Series:
    """c_0 T_0(u) + c_1 T_1(u) + ... in u = (2x - low - high) / (high - low), which runs from -1 to 1 across [low, high]."""

    low: float
    high: float
    coefficients: tuple[float, ...]

    @property
    def name(self) -> str:
        """The series' content, hashed, as a curve's name is: the fit's last bits can differ from one numpy build to the next, and a map is the series it runs."""
        return hashlib.sha256(json.dumps([self.low, self.high, self.coefficients]).encode()).hexdigest()[:16]

    @cached_property
    def slope(self) -> Series:
        """The series' derivative in x, a series over the same pushes: c'_(k-1) = c'_(k+1) + 2k c_k, halved at k = 1, and 2 / (high - low) for u's stretch."""
        derived = [0.0] * (len(self.coefficients) + 1)
        for k in range(len(self.coefficients) - 1, 0, -1):
            derived[k - 1] = derived[k + 1] + 2 * k * self.coefficients[k]
        derived[0] /= 2
        stretch = 2 / (self.high - self.low)
        return Series(self.low, self.high, tuple(coefficient * stretch for coefficient in derived[: max(len(self.coefficients) - 1, 1)]))

    def at(self, x: float) -> float:
        """The series at x, by Clenshaw's recurrence: plain floats, so a state written from it is a float's own spelling."""
        u = (2 * x - (self.low + self.high)) / (self.high - self.low)
        nearer = further = 0.0  # b_{k+1} and b_{k+2}
        for coefficient in reversed(self.coefficients[1:]):
            nearer, further = coefficient + 2 * u * nearer - further, nearer
        return self.coefficients[0] + u * nearer - further


@dataclass(frozen=True)
class Fitted:
    """A series fitted to a curve, and how far the curve's readings lie from it."""

    series: Series
    residual: float  # rms, in the readings' units
    largest: float


def smooth(curve: Curve, degree: int) -> Fitted:
    """The least-squares Chebyshev series of `degree` through the curve, across the pushes it was read at."""
    import warnings

    import numpy
    from numpy.polynomial import Chebyshev

    # [LAW:no-silent-failure] a series through as many readings as it has coefficients passes
    # through every one of them, and its residual of zero would read as a perfect fit.
    if len(curve.values) <= degree + 1:
        raise SmoothError(f"a series of degree {degree} fitted to {len(curve.values)} readings has none left over to measure its residual by; give at least {degree + 2}")
    low, high = curve.values[0], curve.values[-1]
    # In Chebyshev polynomials and not in powers, as `uni.fit` is: on 2901 evenly spaced pushes the
    # powers' normal equations have a condition number of 3e14 at degree 20 and are singular in a
    # float past 60, where these polynomials' least-squares problem has 7 at degree 60 and 11 at 150.
    with warnings.catch_warnings():
        warnings.simplefilter("error", numpy.exceptions.RankWarning)
        try:
            fitted = Chebyshev.fit(curve.values, curve.readings, degree, domain=[low, high])
        except numpy.exceptions.RankWarning as warning:
            raise SmoothError(f"a series of degree {degree} is more than {len(curve.values)} readings from {low:g} to {high:g} can fix: {warning}") from warning
    series = Series(low, high, tuple(float(coefficient) for coefficient in fitted.coef))
    misses = [reading - series.at(value) for value, reading in zip(curve.values, curve.readings)]
    return Fitted(series, math.sqrt(math.fsum(miss * miss for miss in misses) / len(misses)), max(map(abs, misses)))


def jitter(readings: Sequence[float]) -> float:
    """The scatter of evenly spaced readings about the smooth curve under them, from their fourth differences.

    A fourth difference cancels a curve to its fourth derivative times the spacing to the fourth,
    which is nothing at the spacings a curve is read at, and leaves independent scatter of size s
    as a number of variance 70 s^2 (1 + 16 + 36 + 16 + 1). So a fit whose residual comes down to
    this has fitted the curve and not yet its noise.
    """
    differences = readings
    for _ in range(4):
        differences = [after - before for before, after in zip(differences, differences[1:])]
    if not differences:
        raise SmoothError(f"the jitter of a curve is read from its fourth differences, which {len(readings)} readings do not have")
    return math.sqrt(math.fsum(difference * difference for difference in differences) / len(differences) / 70)
