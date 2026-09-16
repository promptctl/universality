"""The response map: the model's answer to a push, fed back as the next push, times a gain."""

from pathlib import Path

import pytest

from uni.cli import EXIT_CONFIG, main
from uni.maps import MapError, ResponseFamily, response_state, response_text
from uni.model import ModelError
from uni.response import response
from uni.steer import Steer, read_direction
from uni.template import load_templates

TEXT = "The meeting moved to Thursday because the room was booked."
FLAGS = ["--map", "response", "--template", "rewrite", "--knob", "formality", "--text", TEXT, "--layer", "23"]


@pytest.fixture(scope="module")
def family(model):
    held = ResponseFamily(model.pinned, load_templates()["rewrite"], TEXT, Steer(read_direction("formality", model.pinned)), 23)
    held.__dict__["model"] = model  # the session's checkpoint, not a second one
    return held


@pytest.mark.parametrize("push, text", [(-8.5, "-8.5000"), (5.33914, "5.3391"), (-0.00004, "0.0000"), (0.0, "0.0000")])
def test_a_push_is_written_to_four_decimals_with_one_zero(push, text):
    # -0.00004 rounds to -0.0, and "-0.0000" beside "0.0000" would be two states for one push.
    assert response_text(push) == text
    assert response_state(text) == float(text)


@pytest.mark.parametrize("state, message", [("-8.5", "write -8.5000"), ("-0.0000", "write 0.0000"), ("nan", "not 'nan'"), ("x", "got 'x'")])
def test_a_state_the_map_would_not_write_is_refused(state, message):
    with pytest.raises(MapError, match=message):
        response_state(state)


def test_a_step_is_the_answer_in_units_of_the_push_times_the_gain(model, family):
    answer = response(model, family.prompt, family.steer, -8.5, 23)
    assert family.at(3.0).step("-8.5000") == response_text(3.0 * answer / family.steer.direction.squared_length)


def test_a_start_too_large_to_read_is_refused_before_a_cell_runs(family):
    with pytest.raises(ModelError, match="rounding could move the answer to a push of 1e\\+06"):
        family.holds(("1000000.0000",))


def test_the_loop_at_gain_three_settles_on_a_period_two_orbit_through_the_top(model, monkeypatch, tmp_path, capsys):
    # Measured (universality-rung2-a27): from the top of the hump the orbit at gain 3 alternates
    # between 5.3391 and -8.1963 from step 9 on, while the fixed point near -0.44 is still stable.
    monkeypatch.setattr("uni.model.Model", lambda pinned: model)
    monkeypatch.setattr("uni.cli.TRAJECTORIES", tmp_path)
    assert main(["loop", *FLAGS, "--value", "3", "--start=-8.5000", "--steps", "20"], {}, Path.cwd()) == 0
    (written,) = tmp_path.iterdir()
    capsys.readouterr()
    assert main(["observe", str(written)], {}, Path.cwd()) == 0
    printed = capsys.readouterr().out
    assert printed.splitlines()[0] == "period 2, entered at step 9"
    assert {line.split()[-1] for line in printed.splitlines()[-4:]} == {"5.339100", "-8.196300"}


@pytest.mark.parametrize("dropped", ["--template", "--knob", "--text", "--layer"])
def test_the_map_names_what_it_was_not_given_before_reading_a_checkpoint(dropped, monkeypatch, capsys):
    monkeypatch.setattr("uni.pinned.load_pinned", lambda: pytest.fail("pinned.toml was read before the flags were"))
    at = FLAGS.index(dropped)
    argv = ["loop", *FLAGS[:at], *FLAGS[at + 2 :], "--value", "3", "--start=-8.5000", "--steps", "1"]
    assert main(argv, {}, Path.cwd()) == EXIT_CONFIG
    assert f"pass {dropped}" in capsys.readouterr().err


def test_the_gain_is_asked_for_rather_than_defaulted(capsys):
    assert main(["loop", *FLAGS, "--start=-8.5000", "--steps", "1"], {}, Path.cwd()) == EXIT_CONFIG
    assert "the response map's parameter is the gain; pass --value" in capsys.readouterr().err
