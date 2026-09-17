"""uni fixed: where a map of numbers holds still, and its slope there, on the map whose answers are known."""

from pathlib import Path

import pytest

from uni.cli import EXIT_CONFIG, main
from uni.fixed import FixedError, crossings, fixed_point, slope
from uni.maps import NUMBERS, Logistic, Numbers


def command(argv):
    return main(argv, {}, Path.cwd())


@pytest.mark.parametrize("r", [1.5, 2.5, 3.2, 3.9])
def test_the_logistic_fixed_point_and_slope_are_the_textbook_ones(r):
    # x* = 1 - 1/r and F'(x*) = 2 - r, so the fixed point gives way at r = 3 exactly.
    numbers = NUMBERS["logistic"]({})
    point = fixed_point(Logistic(r), numbers, 0.1, 0.99)
    assert point == pytest.approx(1 - 1 / r, abs=1e-12)
    assert slope(Logistic(r), numbers, point, 1e-3) == pytest.approx(2 - r, abs=1e-9)


def test_a_bracket_the_map_carries_one_way_is_refused_rather_than_answered_with_an_end():
    with pytest.raises(FixedError, match="carries 0.7 by -0.175 and 0.9 by -0.675; a bracket is two states"):
        fixed_point(Logistic(2.5), NUMBERS["logistic"]({}), 0.7, 0.9)


def test_a_bracket_whose_end_the_map_holds_still_is_refused_rather_than_answered_with_that_end(capsys):
    # 0 is the logistic map's other fixed point: answered, every row would be it, and the slope
    # below it would be read at a state the map cannot hold.
    assert command(["fixed", "--map", "logistic", "--grid", "2.5:3.5:5", "--bracket", "0:0.9", "--step", "0.001"]) == EXIT_CONFIG
    assert "carries 0 by +0 and 0.9 by -0.675; a bracket is two states it carries in opposite directions" in capsys.readouterr().err


def test_a_step_finer_than_the_map_writes_is_refused():
    with pytest.raises(FixedError, match="finer than the map writes its states"):
        slope(Logistic(2.5), Numbers(float, lambda x: "0.6"), 0.6, 1e-3)  # a map that writes every number as one state


def test_crossings_are_placed_between_the_values_either_side():
    assert crossings((1.0, 2.0, 3.0), (0.0, -2.0, 0.0), -1.0) == (1.5, 2.5)
    assert crossings((1.0, 2.0), (-0.5, -1.0), -1.0) == (2.0,)  # landing on the level is passing through it
    assert crossings((1.0, 2.0), (-1.0, -1.0), -1.0) == ()
    assert crossings((1.0, 2.0, 3.0), (-0.9, -1.0, -0.9), -1.0) == ()  # a touch from above and back
    assert crossings((1.0, 2.0, 3.0), (-0.9, -1.0, -1.1), -1.0) == (2.0,)
    assert crossings((1.0, 2.0), (-1.0, -1.25), -1.0) == (1.0,)  # on the level where the grid starts
    assert crossings((1.0, 2.0, 3.0, 4.0), (-0.9, -1.0, -1.0, -1.1), -1.0) == (2.0,)


def test_the_command_finds_the_logistic_flip_at_three(capsys):
    assert command(["fixed", "--map", "logistic", "--grid", "2.5:3.5:5", "--bracket", "0.1:0.99", "--step", "0.001"]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed[3].split() == ["3", "0.6666666666666666", "-1.0000"]
    assert printed[-2:] == ["the slope passes through -1 at 3", "the slope does not pass through +1 on this grid"]


def test_a_map_whose_states_are_texts_has_no_fixed_point_to_find(capsys, monkeypatch):
    monkeypatch.setattr("uni.maps.ModelFamily.model", property(lambda self: pytest.fail("a checkpoint was read")))
    assert command(["fixed", "--map", "model", "--template", "rewrite", "--grid", "0:0:1", "--bracket", "0:1", "--step", "0.1"]) == EXIT_CONFIG
    assert "the model map's are not; the maps whose are: logistic, response, sampled, smooth, noisy" in capsys.readouterr().err


@pytest.mark.parametrize("text", ["1", "1:0", "a:1", "0:nan", "0:1:2"])
def test_a_bracket_that_is_not_one_is_refused_by_the_parser(text, capsys):
    with pytest.raises(SystemExit):
        command(["fixed", "--map", "logistic", "--grid", "3:3:1", f"--bracket={text}", "--step", "0.1"])
    assert "a bracket is LOW:HIGH" in capsys.readouterr().err
