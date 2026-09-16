"""The period of an orbit: how long it takes to come back to a state it has already been in.

The orbit of a deterministic map is exact about this. If a state repeats, the next state
after it is the same next state as last time, and so is every state after that, forever. So
one repeat settles both numbers at once - where the cycle starts and how long it is - and no
window width, threshold, or count of confirming cycles enters into it. That the model's map
is deterministic is what `uni determinism` establishes; the logistic map is by construction.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Cycle:
    """The orbit returns to `onset` every `length` steps, and did so for the rest of the data."""

    length: int
    onset: int  # index in the whole sequence, so onset > burn_in means the transient outlasted it


@dataclass(frozen=True)
class NoCycle:
    """No state repeated, so any period is at least as long as what was looked at.

    At least, and not longer: showing a period of P takes P + 1 states, the one that opens the
    cycle and the one that closes it, so N states without a repeat leave a period of exactly N
    standing as much as any longer one.
    """

    examined: int


@dataclass(frozen=True)
class Contradiction:
    """A state repeated and the orbit then took a different path, so the map is not a function of its state."""

    onset: int
    length: int
    step: int  # the first index holding something other than the state `length` steps before it


# [LAW:types-are-the-program] a period that was not seen has no number to report, because the
# variant that carries a length is the variant that found one.
Period = Cycle | NoCycle | Contradiction


def detect(labels: Sequence[Hashable], burn_in: int = 0) -> Period:
    """The period of `labels` after `burn_in`, or what was seen instead."""
    if burn_in < 0:
        raise ValueError(f"burn_in must not be negative, got {burn_in}")
    seen: dict[Hashable, int] = {}
    for index, label in enumerate(labels[burn_in:], start=burn_in):
        onset = seen.setdefault(label, index)
        if onset == index:
            continue
        # [LAW:no-silent-failure] determinism is the reason one repeat is enough, so it is
        # checked rather than assumed: a cycle that breaks is a map that is not one.
        length = index - onset
        for step in range(onset + length, len(labels)):
            if labels[step] != labels[step - length]:
                return Contradiction(onset=onset, length=length, step=step)
        return Cycle(length=length, onset=onset)
    return NoCycle(examined=max(len(labels) - burn_in, 0))
