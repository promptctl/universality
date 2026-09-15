"""The runner and the trajectory file, on a map simple enough to know every state of."""

from dataclasses import dataclass, field
from itertools import islice

import pytest

from uni.loop import Trajectory, TrajectoryError, orbit, read_trajectory, write_trajectory


@dataclass(frozen=True)
class Append:
    spec: dict = field(default_factory=lambda: {"kind": "append"})

    def step(self, state: str, value: float) -> str:
        return state + str(value)


def trajectory(start="a", steps=3, value=1.5):
    return Trajectory(Append().spec, value, start, tuple(islice(orbit(Append(), value, start), steps)))


def test_orbit_feeds_each_state_back_with_the_value():
    assert trajectory().states == ("a1.5", "a1.51.5", "a1.51.51.5")


def test_zero_steps_is_the_start_alone():
    assert trajectory(steps=0).states == ()


def test_a_trajectory_round_trips_through_its_file(tmp_path):
    original = trajectory(start="héllo\n")
    assert read_trajectory(write_trajectory(original, tmp_path)) == original


def test_the_same_orbit_writes_the_same_bytes_to_the_same_file(tmp_path):
    first = write_trajectory(trajectory(), tmp_path / "one")
    second = write_trajectory(trajectory(), tmp_path / "two")
    assert first.name == second.name
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize("change", [{"start": "b"}, {"steps": 4}, {"value": 2.5}])
def test_different_inputs_name_different_files(change):
    assert trajectory(**change).name != trajectory().name


@pytest.mark.parametrize(
    "text, message",
    [
        ("[]", "must hold a JSON object"),
        ('{"map": {}, "value": 0.0, "start": ""}', "states is missing"),
        ('{"map": {}, "value": 0, "start": "", "states": []}', "value must be a float"),
        ('{"map": {}, "value": 0.0, "start": "", "states": [1]}', "states must all be strings"),
    ],
)
def test_a_file_that_is_not_a_trajectory_is_refused(tmp_path, text, message):
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(TrajectoryError, match=message):
        read_trajectory(path)
