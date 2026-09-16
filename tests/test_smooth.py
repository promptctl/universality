"""The smooth map: a series fitted to a curve `uni response` wrote, read as the response map is, with no model."""

import hashlib
import json
import math
import random
from pathlib import Path

import pytest
from numpy.polynomial import Chebyshev

from uni.cli import EXIT_CONFIG, main
from uni.curve import Curve, CurveError, read_curve, write_curves
from uni.maps import NUMBERS, MapError
from uni.smooth import Series, SmoothError, jitter, smooth

SQUARED_LENGTH = 2.5
PUSHES = [index / 200 for index in range(201)]
# The logistic map, written as a curve: readings of x (1 - x) |v|^2, which the smooth map divides
# by |v|^2 and multiplies by the gain, so at gain r the fitted map is x -> r x (1 - x) itself, and
# its superstable values are the ones tests/test_cascade.py takes from mpmath.
LOGISTIC = {7: [SQUARED_LENGTH * x * (1 - x) for x in PUSHES]}
GRIDS = ("3.2355:3.2365:11", "3.4981:3.4991:11", "3.5544:3.5549:11", "3.56655:3.56675:11", "3.5692:3.5693:11")


def command(argv):
    return main(argv, {}, Path.cwd())


@pytest.fixture
def logistic(tmp_path):
    return write_curves({"what": "the logistic map"}, PUSHES, LOGISTIC, SQUARED_LENGTH, tmp_path)


def flags(curve, degree=2):
    return ["--map", "smooth", "--curve", str(curve), "--layer", "7", "--degree", str(degree)]


def test_the_logistic_map_fitted_as_a_curve_has_the_logistic_cascade(logistic, capsys):
    argv = ["cascade", *flags(logistic), "--critical", "0.5", "--period", "2"] + [flag for grid in GRIDS for flag in ("--grid", grid)]
    assert command(argv) == 0
    printed = capsys.readouterr().out.splitlines()
    assert [float(line.split()[1]) for line in printed[1:6]] == pytest.approx([3.23606797749979, 3.4985616993277, 3.55464086276882, 3.56666737985627, 3.56924353163711], abs=2e-9)
    assert [float(line.split(": ")[1].split()[0]) for line in printed[6:9]] == pytest.approx([4.6808, 4.6630, 4.6684], abs=1e-4)
    assert [float(line.split(": ")[1].split()[0]) for line in printed[9:]] == pytest.approx([-2.6547, -2.5318, -2.5087, -2.5041], abs=1e-4)


def test_its_top_is_where_the_logistic_map_s_is(logistic, capsys):
    assert command(["critical", *flags(logistic), "--value", "3.5", "--grid", "0.4:0.62:23"]) == 0
    assert float(capsys.readouterr().out.split()[-1]) == pytest.approx(0.5, abs=1e-12)


def test_a_curve_is_named_by_its_bytes_and_the_map_by_that_name_and_its_series_bits(logistic):
    assert logistic.stem == hashlib.sha256(logistic.read_bytes()).hexdigest()[:16]
    from uni.cli import MAPS, build_parser

    args = build_parser().parse_args(["loop", *flags(logistic, 5), "--start", "0.5", "--steps", "1", "--value", "3"])
    family = MAPS["smooth"].build(args, (3.0,))
    assert family.spec == {"kind": "smooth", "curve": logistic.stem, "layer": 7, "degree": 5, "series": family.series.name}
    changed = Series(family.series.low, family.series.high, (*family.series.coefficients[:-1], math.nextafter(family.series.coefficients[-1], 1)))
    assert changed.name != family.series.name


def test_the_series_is_numpy_s_chebyshev_series_evaluated_in_plain_floats():
    coefficients = tuple(random.Random(1).uniform(-1, 1) for _ in range(40))
    series, reference = Series(-19.0, 10.0, coefficients), Chebyshev(coefficients, domain=[-19.0, 10.0])
    for x in (-19.0, -8.614, 0.0, 3.3, 10.0):
        assert type(series.at(x)) is float and series.at(x) == pytest.approx(float(reference(x)), abs=1e-12)


def test_a_fit_to_a_noisy_curve_comes_down_to_the_noise_and_the_jitter_says_how_much_that_is():
    # A hump with scatter of a known size: the fourth differences find that size, and a series of
    # enough degree leaves a residual of it and no more.
    noise = random.Random(7)
    values = [-19 + 29 * index / 2900 for index in range(2901)]
    readings = [-3 * math.exp(-(((x + 8.6) / 6) ** 2)) + 0.2 * math.sin(x) + noise.gauss(0, 1e-3) for x in values]
    assert jitter(readings) == pytest.approx(1e-3, rel=0.05)
    fitted = smooth(Curve("noisy", 23, tuple(values), tuple(readings), 8.9), 40)
    assert fitted.residual == pytest.approx(1e-3, rel=0.05) and fitted.largest > fitted.residual


def test_an_orbit_that_leaves_the_pushes_the_series_was_fitted_across_is_refused(logistic, capsys):
    # At r = 4.5 the logistic map carries 0.5 to 1.125, past the last push on the curve.
    assert command(["loop", *flags(logistic), "--start", "0.5", "--steps", "3", "--value", "4.5"]) == EXIT_CONFIG
    assert "a push of 1.12499999" in (err := capsys.readouterr().err) and "lies outside the curve's pushes, 0 to 1" in err


def test_a_start_outside_the_pushes_is_refused_before_a_step(logistic, capsys):
    assert command(["loop", *flags(logistic), "--start=-0.5", "--steps", "1", "--value", "3"]) == EXIT_CONFIG
    assert "a push of -0.5 lies outside the curve's pushes" in capsys.readouterr().err


@pytest.mark.parametrize("state, message", [("0.50", "write 0.5, not '0.50'"), ("nan", "a finite push"), ("x", "a number, got 'x'")])
def test_a_state_is_a_push_spelled_as_its_own_shortest_text(state, message):
    with pytest.raises(MapError, match=message):
        NUMBERS["smooth"]({}).read(state)


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--decimals", "6"], "--decimals does not describe the smooth map, which reads --curve, --layer, --degree"),
        ([], "the smooth map's parameter is the gain; pass --value"),
    ],
)
def test_the_smooth_map_refuses_what_it_does_not_read_and_needs_its_gain(logistic, capsys, extra, message):
    argv = ["loop", *flags(logistic), "--start", "0.5", "--steps", "1", *extra] + ([] if not extra else ["--value", "3"])
    assert command(argv) == EXIT_CONFIG
    assert message in capsys.readouterr().err


def test_a_smooth_map_without_its_curve_says_what_it_needs(capsys):
    assert command(["loop", "--map", "smooth", "--layer", "7", "--start", "0.5", "--steps", "1", "--value", "3"]) == EXIT_CONFIG
    assert "a series of some degree fitted to a curve's readings at a layer; pass --curve, --degree" in capsys.readouterr().err


def test_the_response_map_refuses_a_curve(capsys):
    argv = ["loop", "--map", "response", "--template", "rewrite", "--knob", "formality", "--text", "a", "--layer", "23", "--curve", "c.json", "--start", "0", "--steps", "1", "--value", "1"]
    assert command(argv) == EXIT_CONFIG
    assert "--curve does not describe the response map" in capsys.readouterr().err


def test_the_command_reports_each_degree_s_residual_beside_the_curve_s_jitter(logistic, capsys):
    assert command(["smooth", "--curve", str(logistic), "--layer", "7", "--degree", "2", "--degree", "5"]) == 0
    first, *degrees = capsys.readouterr().out.splitlines()
    assert first.startswith(f"curve {logistic.stem} at layer 7: 201 readings from 0 to 1, jitter ")
    assert [line.split(":")[0] for line in degrees] == ["degree    2", "degree    5"]
    assert all(float(line.split()[4].rstrip(",")) < 1e-14 for line in degrees)


def test_a_degree_the_readings_cannot_fix_is_refused():
    with pytest.raises(SmoothError, match="degree 5 fitted to 6 readings has none left over"):
        smooth(Curve("short", 7, tuple(PUSHES[:6]), tuple(LOGISTIC[7][:6]), 1.0), 5)
    # Rising, but bunched so that the pushes cannot tell a degree-12 series' terms apart.
    values = tuple([index * 1e-3 for index in range(12)] + [1000.0 + index for index in range(3)])
    with pytest.raises(SmoothError, match="a series of degree 12 is more than 15 readings from 0 to 1002 can fix"):
        smooth(Curve("bunched", 7, values, tuple(float(index % 3) for index in range(15)), 1.0), 12)


def curve_file(tmp_path, **changes):
    body = {"described": {}, "squared_length": 2.5, "values": [0.0, 0.5, 1.0], "layers": {"7": [0.0, 1.0, 0.0]}} | changes
    path = tmp_path / "curve.json"
    path.write_text(json.dumps(body))
    return path


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"layers": {"23": [0.0, 1.0, 0.0]}}, "holds the curve at layer 23, not at 7"),
        ({"layers": {"7": [0.0, 1.0]}}, "one reading at layer 7 for each of its 3 pushes"),
        ({"values": [], "layers": {"7": []}}, "holds no pushes"),
        ({"values": [0.0, 1.0, 0.5]}, "the pushes in .* must rise"),
        ({"values": [0.0, 0.5, 1]}, "every push in .* must be a finite number"),
        ({"layers": {"7": [0.0, None, 0.0]}}, "every reading in .* must be a finite number"),
        ({"squared_length": -1.0}, "squared_length in .* must be above zero, got -1.0"),
        ({"squared_length": 2}, "squared_length must be a float, got 2"),
    ],
)
def test_a_curve_file_that_is_not_a_curve_is_refused(tmp_path, changes, message):
    with pytest.raises(CurveError, match=message):
        read_curve(curve_file(tmp_path, **changes), 7)


def test_a_curve_read_on_a_falling_grid_is_written_rising(tmp_path):
    # `uni response --grid 1:-1:3` reads the pushes in that order; the file holds them as a curve.
    path = write_curves({}, (1.0, 0.0, -1.0), {7: (0.25, 0.5, 0.75), 9: (1.0, 2.0, 3.0)}, 2.5, tmp_path)
    assert (read_curve(path, 7).values, read_curve(path, 7).readings, read_curve(path, 9).readings) == ((-1.0, 0.0, 1.0), (0.75, 0.5, 0.25), (3.0, 2.0, 1.0))


def test_a_curve_file_that_is_not_there_or_not_json_is_refused(tmp_path):
    with pytest.raises(CurveError, match="cannot be read"):
        read_curve(tmp_path / "absent.json", 7)
    (tmp_path / "half.json").write_text('{"values": [')
    with pytest.raises(CurveError, match="is not JSON"):
        read_curve(tmp_path / "half.json", 7)
