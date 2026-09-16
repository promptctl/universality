"""The period detector, on sequences whose period is known because they were built with one."""

import pytest

from uni.cli import verdict
from uni.period import Contradiction, Cycle, NoCycle, detect


@pytest.mark.parametrize(
    "labels, expected",
    [
        ("aaaaaa", Cycle(length=1, onset=0)),
        ("abababab", Cycle(length=2, onset=0)),
        ("abcabcabc", Cycle(length=3, onset=0)),
        ("abcdef", NoCycle(examined=6)),
    ],
    ids=["period 1", "period 2", "period 3", "no period within the horizon"],
)
def test_the_period_of_a_sequence_built_with_one(labels, expected):
    assert detect(labels) == expected


def test_a_transient_that_outlasts_the_burn_in_is_reported_with_where_it_ended():
    # Three steps of transient and a burn-in of one, so the burn-in did not cover it.
    assert detect("xyzababab", burn_in=1) == Cycle(length=2, onset=3)


def test_a_period_longer_than_the_window_is_not_reported_as_a_period():
    # Ten steps of a cycle of twenty: nothing has come back round yet, and saying otherwise
    # would be inventing the one number this detector exists to be trusted about.
    assert detect("abcdefghij") == NoCycle(examined=10)


@pytest.mark.parametrize("steps", range(1, 12))
def test_an_orbit_that_never_repeats_never_reports_a_period(steps):
    assert detect([f"state {n}" for n in range(steps)]) == NoCycle(examined=steps)


def test_a_state_that_returns_and_then_goes_elsewhere_is_not_a_period():
    # Only a map that is not a function of its state can do this, so it is neither a cycle
    # nor an absence of one, and reporting either would bury a broken determinism gate.
    # The step reported is the one holding the state that broke the cycle: index 5, the 'c',
    # and not the on-cycle index it was compared against.
    contradiction = detect("ababac")
    assert contradiction == Contradiction(onset=0, length=2, step=5)
    assert "ababac"[contradiction.step] == "c"


def test_the_burn_in_is_not_searched():
    # The period-1 run of a's is passed over, and what follows it never repeats.
    assert detect("aaabcdef", burn_in=3) == NoCycle(examined=5)


def test_a_repeat_that_straddles_the_burn_in_is_not_a_repeat():
    # 'abc' appears twice, but only the second is looked at, so nothing has come back.
    assert detect("abcabc", burn_in=3) == NoCycle(examined=3)


def test_a_window_that_shows_no_repeat_leaves_a_period_its_own_length_standing():
    # Showing a period of 3 takes 4 states, so these 3 rule out nothing at all: the orbit they
    # come from has period 3 exactly. The sentence the command prints has to say "at least".
    assert "at least 3" in verdict(detect("abcabc", burn_in=3))


def test_a_burn_in_past_the_end_examines_nothing():
    assert detect("abc", burn_in=10) == NoCycle(examined=0)


def test_a_negative_burn_in_is_refused():
    with pytest.raises(ValueError, match="must not be negative"):
        detect("abc", burn_in=-1)


def test_the_labels_can_be_anything_hashable():
    # The detector compares states; hashing them first is the caller's choice, not its own.
    import hashlib

    states = ["one", "two", "one", "two"]
    hashed = [hashlib.sha256(state.encode()).hexdigest() for state in states]
    assert detect(hashed) == detect(states) == Cycle(length=2, onset=0)
