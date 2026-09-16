"""uni cascade: the superstable values of a cascade and the ratios of their spacings, on the map whose are known."""

import math
from pathlib import Path

import pytest

from uni import cascade
from uni.cascade import Reading, amplification, evaluated, growths, nearest, quotients, ratios, returns
from uni.fit import Estimate, FitError, crossing, fit
from uni.cli import EXIT_CONFIG, main
from uni.maps import NUMBERS, Logistic

# The logistic map's superstable r for periods 2 to 32, from its own orbit of 0.5 solved to 40 digits
# with mpmath, independently of this code. Their spacing ratios are 4.6808, 4.6630, 4.6684.
SUPERSTABLE = {2: 3.23606797749979, 4: 3.4985616993277, 8: 3.55464086276882, 16: 3.56666737985627, 32: 3.56924353163711}
# F^(p/2)(0.5) - 0.5 at each of those values, from the same orbits in mpmath: the distance from the
# top of the cycle's point nearest it. Their ratios are -2.6547448, -2.5318377, -2.5087182, -2.5041128.
NEAREST = {2: 0.3090169943749474, 4: -0.1164017695468324, 8: 0.04597521058174422, 16: -0.01832617573366167, 32: 0.007318430628499477}
# sqrt(sum_k prod_(j=k..p-1) F'(x_j)^2) along the same cycles in mpmath: how far unit noise at every
# step moves where the cycle returns. With NEAREST, the growths 6.8839697, 6.660634, 6.627796, 6.620746.
NOISE = {2: 2.23606797749979, 4: 5.798306550714591, 8: 15.25389979049446, 16: 40.29935935822024, 32: 106.5494423745227}
GRIDS = {2: "3.2355:3.2365:11", 4: "3.4981:3.4991:11", 8: "3.5544:3.5549:11", 16: "3.56655:3.56675:11", 32: "3.5692:3.5693:11"}


def command(argv):
    return main(argv, {}, Path.cwd())


def read(period, grid, critical="0.5"):
    first, last, count = grid.split(":")
    values = [float(first) + (float(last) - float(first)) * index / (int(count) - 1) for index in range(int(count))]
    returned = [returns(Logistic(value), NUMBERS["logistic"]({}), critical, period) for value in values]
    zero = crossing(cascade.superstable(values, returned, period), 0)
    return zero, nearest(values, returned, period, zero)


def superstable(period, grid, critical="0.5"):
    return read(period, grid, critical)[0]


@pytest.mark.parametrize("period", SUPERSTABLE)
def test_the_logistic_superstable_values_are_the_known_ones(period):
    zero = superstable(period, GRIDS[period])
    assert zero.value == pytest.approx(SUPERSTABLE[period], abs=2e-9)
    # And the error it reports covers how far off it is: the parabola follows the return's curve,
    # which a line would not, and a line's error would not cover what that curve does to it.
    assert abs(zero.value - SUPERSTABLE[period]) < 3 * zero.error


@pytest.mark.parametrize("period", NEAREST)
def test_the_cycle_s_nearest_point_to_the_top_is_the_known_one_and_its_error_covers_it(period):
    distance = read(period, GRIDS[period])[1].estimate
    assert distance.value == pytest.approx(NEAREST[period], abs=4e-9)
    assert abs(distance.value - NEAREST[period]) < 3 * distance.error


def test_the_nearest_points_ratios_run_to_minus_alpha():
    found = quotients(tuple(read(period, GRIDS[period])[1].estimate for period in NEAREST))
    assert [estimate.value for estimate in found] == pytest.approx([-2.6547448, -2.5318377, -2.5087182, -2.5041128], abs=3e-6)


@pytest.mark.parametrize("period", NOISE)
def test_the_noise_a_cycle_carries_is_the_known_amount_and_its_error_covers_it(period):
    zero = superstable(period, GRIDS[period])
    first, last, count = GRIDS[period].split(":")
    values = [float(first) + (float(last) - float(first)) * index / (int(count) - 1) for index in range(int(count))]
    noise = evaluated(values, [amplification(Logistic(value), NUMBERS["logistic"]({}), "0.5", period) for value in values], zero).estimate
    assert noise.value == pytest.approx(NOISE[period], rel=1e-6)
    assert abs(noise.value - NOISE[period]) < 3 * noise.error


def test_unit_noise_after_one_step_is_one_and_after_two_is_carried_by_the_slope_between():
    # From the top 0.5 at r = 3: x_1 = 0.75, where the slope is 3 (1 - 1.5) = -1.5.
    assert amplification(Logistic(3.0), NUMBERS["logistic"]({}), "0.5", 1) == 1.0
    assert amplification(Logistic(3.0), NUMBERS["logistic"]({}), "0.5", 2) == pytest.approx(math.sqrt(1 + 1.5**2))


def test_a_growth_is_the_noise_ratio_times_the_distance_ratio_with_four_relative_errors():
    zero, later = Estimate(3.5, 0.01), Estimate(3.55, 0.01)
    noises = (Reading(2.0, 0.02, 0.0, zero), Reading(5.0, 0.1, 0.0, later))
    (found,) = growths(noises, (Reading(0.3, 0.003, 0.0, zero), Reading(-0.12, 0.0024, 0.0, later)))
    assert (found.value, found.error) == pytest.approx((6.25, 6.25 * math.sqrt(2 * 0.01**2 + 2 * 0.02**2)))


def test_a_superstable_value_s_error_reaches_a_growth_once_through_the_difference_of_its_readings_relative_slopes():
    # At the first value both readings move by twice themselves per unit of gain, so its error
    # moves the growth not at all; at the second, the noise by twice and the distance by once.
    zero, later = Estimate(3.5, 0.01), Estimate(3.55, 0.01)
    noises = (Reading(2.0, 0.0, 4.0, zero), Reading(5.0, 0.0, 10.0, later))
    (found,) = growths(noises, (Reading(0.3, 0.0, 0.6, zero), Reading(-0.12, 0.0, -0.12, later)))
    assert (found.value, found.error) == pytest.approx((6.25, 6.25 * 0.01))


def test_a_quotient_s_error_is_its_two_values_relative_errors_added_in_quadrature():
    (found,) = quotients((Estimate(-3.0, 0.03), Estimate(1.5, 0.02)))
    assert (found.value, found.error) == pytest.approx((-2.0, 2.0 * math.hypot(0.01, 0.02 / 1.5)))


def test_a_critical_point_off_by_a_little_moves_each_value_by_that_over_how_steeply_the_return_crosses():
    # Why the top is measured before the cascade is read from it: an x_c off by 1e-4 moves the
    # period-2 value by 1e-4 over how steeply the return crosses zero there, about 0.35 per unit of r.
    moved = superstable(2, GRIDS[2], "0.5001").value - superstable(2, GRIDS[2]).value
    assert abs(moved) == pytest.approx(2.9e-4, rel=0.05)


@pytest.mark.parametrize("period, grids", [("2", (GRIDS[2], GRIDS[2])), ("4", (GRIDS[2],))])
def test_a_grid_holding_a_shorter_cycle_s_value_is_refused_rather_than_read_as_its_own(capsys, period, grids):
    # At the period-2 value the top returns after 4 steps as well, so a period-4 grid placed there
    # would find it again, and a spacing of zero would come out as a ratio like any other.
    argv = ["cascade", "--map", "logistic", "--critical", "0.5", "--period", period] + [flag for text in grids for flag in ("--grid", text)]
    assert command(argv) == EXIT_CONFIG
    assert "holds a superstable value of period 2: the orbit of the top closes after 2 steps there, and so after 4 as well" in capsys.readouterr().err


def test_the_returns_of_a_period_are_those_after_each_number_of_steps_dividing_it():
    assert sorted(returns(Logistic(3.5), NUMBERS["logistic"]({}), "0.5", 12)) == [1, 2, 3, 4, 6, 12]
    assert list(returns(Logistic(3.5), NUMBERS["logistic"]({}), "0.5", 1)) == [1]


def test_the_command_reads_the_logistic_cascade_and_its_ratios(capsys):
    argv = ["cascade", "--map", "logistic", "--critical", "0.5", "--period", "2"]
    for period in SUPERSTABLE:
        argv += ["--grid", GRIDS[period]]
    assert command(argv) == 0
    printed = capsys.readouterr().out.splitlines()
    assert [float(line.split()[1]) for line in printed[1:6]] == pytest.approx(list(SUPERSTABLE.values()), abs=2e-9)
    assert [float(line.split()[4]) for line in printed[1:6]] == pytest.approx(list(NEAREST.values()), abs=4e-9)
    spacings = printed[6:9]
    assert [line.split(": ")[0] for line in spacings] == [f"spacing ratio over periods {p}" for p in ("2, 4, 8", "4, 8, 16", "8, 16, 32")]
    assert [float(line.split(": ")[1].split()[0]) for line in spacings] == pytest.approx([4.6808, 4.6630, 4.6684], abs=1e-4)
    assert [float(line.split()[6]) for line in printed[1:6]] == pytest.approx(list(NOISE.values()), rel=1e-6)
    nearests, noises = printed[9:13], printed[13:]
    assert [line.split(": ")[0] for line in nearests] == [f"nearest-point ratio over periods {p}" for p in ("2, 4", "4, 8", "8, 16", "16, 32")]
    assert [float(line.split(": ")[1].split()[0]) for line in nearests] == pytest.approx([-2.6547, -2.5318, -2.5087, -2.5041], abs=1e-4)
    assert [line.split(": ")[0] for line in noises] == [f"noise growth over periods {p}" for p in ("2, 4", "4, 8", "8, 16", "16, 32")]
    assert [float(line.split(": ")[1].split()[0]) for line in noises] == pytest.approx([6.8839697, 6.660634, 6.627796, 6.620746], abs=1e-5)


def test_a_cascade_from_an_odd_period_has_no_nearest_point_on_its_first_row(capsys):
    assert command(["cascade", "--map", "logistic", "--critical", "0.5", "--period", "1", "--grid", "1.9:2.1:11", "--grid", GRIDS[2], "--grid", GRIDS[4]]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert [len(line.split()) for line in printed[1:4]] == [4, 8, 8]
    assert printed[-2].startswith("nearest-point ratio over periods 2, 4: -2.654744")
    assert printed[-1].startswith("noise growth over periods 2, 4: 6.88396")


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
        # Not exactly repeated values alone: rounding leaves the elimination a pivot a hair off zero.
        ([3.1, 3.3, 3.7] * 3, [1.0, -0.5, 2.0, 1.1, -0.4, 2.1, 0.9, -0.6, 1.9], 3, "needs readings at 4 distinct values at least, and these are at 3"),
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
