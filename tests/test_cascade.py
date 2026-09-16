"""uni cascade: the superstable values of a cascade and the ratios of their spacings, on the map whose are known."""

import math
from pathlib import Path

import pytest

from uni.cascade import ratios, returned
from uni.fit import Estimate, FitError, crossing, fit
from uni.cli import EXIT_CONFIG, main
from uni.maps import NUMBERS, Logistic

# The logistic map's superstable r for periods 2 to 32, from its own orbit of 0.5 solved to 40 digits
# with mpmath, independently of this code. Their spacing ratios are 4.6808, 4.6630, 4.6684.
SUPERSTABLE = {2: 3.23606797749979, 4: 3.4985616993277, 8: 3.55464086276882, 16: 3.56666737985627, 32: 3.56924353163711}
GRIDS = {2: "3.2355:3.2365:11", 4: "3.4981:3.4991:11", 8: "3.5544:3.5549:11", 16: "3.56655:3.56675:11", 32: "3.5692:3.5693:11"}


def command(argv):
    return main(argv, {}, Path.cwd())


def superstable(period, grid, critical="0.5"):
    first, last, count = grid.split(":")
    values = [float(first) + (float(last) - float(first)) * index / (int(count) - 1) for index in range(int(count))]
    return crossing(fit(values, [returned(Logistic(value), NUMBERS["logistic"]({}), critical, period) for value in values], 2), 0)


@pytest.mark.parametrize("period", SUPERSTABLE)
def test_the_logistic_superstable_values_are_the_known_ones(period):
    zero = superstable(period, GRIDS[period])
    assert zero.value == pytest.approx(SUPERSTABLE[period], abs=2e-9)
    # And the error it reports covers how far off it is: the parabola follows the return's curve,
    # which a line would not, and a line's error would not cover what that curve does to it.
    assert abs(zero.value - SUPERSTABLE[period]) < 3 * zero.error


def test_a_critical_point_off_by_a_little_moves_each_value_by_that_over_how_steeply_the_return_crosses():
    # Why the top is measured before the cascade is read from it: an x_c off by 1e-4 moves the
    # period-2 value by 1e-4 over how steeply the return crosses zero there, about 0.35 per unit of r.
    moved = superstable(2, GRIDS[2], "0.5001").value - superstable(2, GRIDS[2]).value
    assert abs(moved) == pytest.approx(2.9e-4, rel=0.05)


def test_the_command_reads_the_logistic_cascade_and_its_ratios(capsys):
    argv = ["cascade", "--map", "logistic", "--critical", "0.5", "--period", "2"]
    for period in SUPERSTABLE:
        argv += ["--grid", GRIDS[period]]
    assert command(argv) == 0
    printed = capsys.readouterr().out.splitlines()
    assert [float(line.split()[1]) for line in printed[1:6]] == pytest.approx(list(SUPERSTABLE.values()), abs=2e-9)
    assert [line.split(": ")[0] for line in printed[6:]] == [f"spacing ratio over periods {p}" for p in ("2, 4, 8", "4, 8, 16", "8, 16, 32")]
    assert [float(line.split(": ")[1].split()[0]) for line in printed[6:]] == pytest.approx([4.6808, 4.6630, 4.6684], abs=1e-4)


def test_a_line_through_exact_readings_crosses_where_it_should_with_no_error():
    line = fit([1.0, 2.0, 3.0, 4.0], [-3.0, -1.0, 1.0, 3.0], 1)
    assert crossing(line, 0) == Estimate(2.5, 0.0) and line.scatter == 0.0


def test_scatter_about_a_line_gives_its_zero_the_textbook_error():
    values, readings = [1.0, 2.0, 3.0, 4.0, 5.0], [-2.1, -0.9, 0.2, 0.8, 2.1]
    zero = crossing(fit(values, readings, 1), 0)
    # The textbook line, written out: slope Sxy/Sxx, zero at mean_x - mean_y/slope, and
    # error s/|slope| sqrt(1/n + (zero - mean_x)^2/Sxx) with s^2 the residual sum over n - 2.
    mean_x, mean_y = 3.0, sum(readings) / 5
    sxx = sum((x - mean_x) ** 2 for x in values)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(values, readings)) / sxx
    at = mean_x - mean_y / slope
    s2 = sum((y - mean_y - slope * (x - mean_x)) ** 2 for x, y in zip(values, readings)) / 3
    assert zero.value == pytest.approx(at, abs=1e-12)
    assert zero.error == pytest.approx(math.sqrt(s2) / abs(slope) * math.sqrt(1 / 5 + (at - mean_x) ** 2 / sxx), rel=1e-9)


def test_the_top_of_a_cubic_is_where_its_slope_crosses_zero():
    # y = -(x - 2)^2 + 0.1 (x - 2)^3 turns at x = 2 and again at 2 + 2/0.3, which is off this grid.
    values = [1.5 + 0.1 * index for index in range(11)]
    top = crossing(fit(values, [-((x - 2) ** 2) + 0.1 * (x - 2) ** 3 for x in values], 3), 1)
    assert top.value == pytest.approx(2.0, abs=1e-12) and top.error < 1e-12


@pytest.mark.parametrize(
    "values, readings, degree, message",
    [
        ([1.0, 2.0], [-1.0, 1.0], 1, "degree 1 fitted to 2 readings has none left over"),
        ([1.0, 1.0, 1.0], [-1.0, 0.0, 1.0], 1, "needs readings at 2 distinct values at least, and these are at 1"),
        ([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], 1, "crosses zero 0 times"),
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 1, "crosses zero 0 times between 1 and 3"),
    ],
)
def test_readings_that_do_not_place_one_crossing_on_their_grid_are_refused(values, readings, degree, message):
    with pytest.raises(FitError, match=message):
        crossing(fit(values, readings, degree), 0)


def test_the_command_finds_the_logistic_top_at_one_half(capsys):
    assert command(["critical", "--map", "logistic", "--value", "3.5", "--grid", "0.4:0.62:23"]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("the map at 3.5 turns at 0.5 +- ")
    assert float(printed.split()[-1]) == pytest.approx(0.5, abs=1e-12)  # the fit's own float, written as the map writes it


def test_a_ratio_carries_the_error_of_the_value_its_two_spacings_share():
    # Spacings 2 and 1; the middle value's error moves both, in opposite directions.
    (ratio,) = ratios([Estimate(0.0, 0.0), Estimate(2.0, 0.1), Estimate(3.0, 0.0)])
    assert ratio.value == 2.0
    assert ratio.error == pytest.approx(2.0 * 0.1 * (1 / 2 + 1 / 1))


def test_a_map_whose_states_are_texts_has_no_cascade_to_read(capsys, monkeypatch):
    monkeypatch.setattr("uni.maps.ModelFamily.model", property(lambda self: pytest.fail("a checkpoint was read")))
    argv = ["cascade", "--map", "model", "--template", "rewrite", "--critical", "x", "--period", "1", "--grid", "0:0:1"]
    assert command(argv) == EXIT_CONFIG
    assert "uni cascade reads a map whose states are numbers, and the model map's are not" in capsys.readouterr().err


def test_a_critical_point_the_map_would_not_write_is_refused(capsys):
    assert command(["cascade", "--map", "logistic", "--critical", "0.50", "--period", "2", "--grid", GRIDS[2]]) == EXIT_CONFIG
    assert "write 0.5, not '0.50'" in capsys.readouterr().err
