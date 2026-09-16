"""The logistic map, the one map here whose right answer is known before the code runs.

Every other test in this repo checks that a brick does what this codebase says it does. These
check that the bricks, composed, reproduce a result that was published in 1978 - the period
doubling that the whole project exists to look for somewhere else.
"""

import subprocess
import sys
from itertools import islice
from pathlib import Path

import pytest

from uni.cli import EXIT_CONFIG, MAP_OPTIONS, main
from uni.loop import Trajectory, orbit, read_trajectory, write_trajectory
from uni.maps import Logistic, MapError, logistic_state
from uni.observe import ObserveError, Step, Value, readings
from uni.period import Cycle, NoCycle, detect


def states(r, start="0.5", steps=400):
    """The orbit of r from `start`, through the same runner the model's map goes through."""
    return tuple(islice(orbit(Logistic(r), start), steps))


@pytest.mark.parametrize(
    "r, length",
    [(2.8, 1), (3.2, 2), (3.5, 4), (3.55, 8)],
    ids=["period 1", "period 2", "period 4", "period 8"],
)
def test_the_cascade_doubles_where_feigenbaum_said_it_does(r, length):
    # The orbit is exactly periodic, not nearly: float64 arithmetic lands on the cycle and stays
    # there, so the detector's own exactness is what reports it, with nothing rounded to help.
    period = detect(states(r))
    assert isinstance(period, Cycle), period
    assert period.length == length


def test_past_the_cascade_no_period_is_claimed():
    # r = 3.9 is chaotic. A thousand steps show no repeat, and the detector says only that -
    # the one place a made-up number would be indistinguishable from a discovery.
    assert detect(states(3.9, steps=1000)) == NoCycle(examined=1000)


def test_a_step_is_the_textbook_arithmetic():
    assert Logistic(3.2).step("0.5") == "0.8"
    assert Logistic(2.0).step("0.25") == "0.375"


def test_a_state_is_the_text_that_reads_back_as_itself():
    # The detector compares the states themselves, so a state has to be the one spelling of its
    # number: a state that read back as a different float would be a cycle that never closes.
    for state in states(3.9, steps=50):
        assert repr(float(state)) == state


def test_the_map_carries_its_interval_into_itself():
    # r = 4 is the edge: the peak of r x (1 - x) is exactly 1, and 1 steps to 0.
    assert all(0 <= float(state) <= 1 for state in states(4.0, steps=200))
    assert Logistic(4.0).step("0.5") == "1.0"
    assert Logistic(4.0).step("1.0") == "0.0"


def test_the_orbit_is_fixed_by_r_and_the_start_alone():
    assert states(3.5, steps=20) == states(3.5, steps=20)


def test_the_spec_names_the_map_and_nothing_r_already_says():
    # r is the trajectory's value, written once, where a steering value is written.
    assert Logistic(3.2).spec == {"kind": "logistic"}


@pytest.mark.parametrize("r", [-0.5, 4.5, float("1e999"), float("-1e999"), float("nan")])
def test_an_r_that_carries_the_interval_out_of_itself_is_refused(r):
    with pytest.raises(MapError, match="r must be in 0..4"):
        Logistic(r)


# The infinities are written as powers here and above, as they are in tests/test_steer.py: the
# run host's name is a short common word, and no tracked file may carry one. 1e999 is the same
# float, and a state naming it is refused by the range check either way it is spelled.
@pytest.mark.parametrize("state", ["abc", "", "1.5", "-0.1", "nan", "1e999", "-1e999", "0.5 0.5"])
def test_a_state_this_map_cannot_step_is_refused(state):
    with pytest.raises(MapError, match="a logistic state is a number in 0..1"):
        Logistic(3.2).step(state)


@pytest.mark.parametrize("state", ["0.50", " 0.5 ", "+0.5", "5e-1", ".5"])
def test_a_second_spelling_of_a_number_this_map_holds_is_refused(state):
    # Each of these is 0.5, and none is what the map writes for it. Accepted, one of them as a
    # --start would be a state the detector counts as new and the plot draws on top of the state
    # it repeats: a fixed point reported as a transient, with nothing anywhere saying so.
    with pytest.raises(MapError, match="shortest text that reads back as its number: write 0.5"):
        Logistic(3.2).step(state)


def test_a_start_the_map_could_have_written_is_the_one_it_takes(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr("uni.cli.TRAJECTORIES", tmp_path)
    assert command(["loop", "--map", "logistic", "--start", "0.50", "--steps", "1", "--value", "3.2"]) == EXIT_CONFIG
    assert "write 0.5, not '0.50'" in capsys.readouterr().err


def command(argv):
    return main(argv, {}, Path.cwd())


def loop(tmp_path, monkeypatch, r, steps):
    monkeypatch.setattr("uni.cli.TRAJECTORIES", tmp_path)
    assert command(["loop", "--map", "logistic", "--start", "0.5", "--steps", str(steps), "--value", str(r)]) == 0
    (path,) = tmp_path.glob("*.json")
    return path


def test_the_command_runs_the_map_and_writes_the_orbit(tmp_path, monkeypatch, capsys):
    path = loop(tmp_path, monkeypatch, 3.2, 4)
    assert "0.7995392" in capsys.readouterr().out
    assert read_trajectory(path) == Trajectory({"kind": "logistic"}, 3.2, "0.5", ("0.8", "0.512", "0.7995392", "0.512884056522752"))


def test_the_command_reads_the_orbit_back_with_the_same_observables(tmp_path, monkeypatch, capsys):
    path = loop(tmp_path, monkeypatch, 3.2, 60)
    capsys.readouterr()
    assert command(["observe", str(path)]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed[0] == "period 2, entered at step 33"
    # Length is what any map's states answer; the model's observables are absent rather than
    # refused, because a number has no prompt behind it.
    assert printed[2].split() == ["step", "state", "length", "x"]
    # The period is visible in the column it was designed to be visible in.
    assert [row.split()[1] for row in printed[-4:]] == ["34", "35", "34", "35"]


def test_the_state_is_the_observable():
    # Nothing to derive: the return map of a map whose states are numbers is the map itself drawn,
    # which is what makes this orbit's plot checkable against a published one.
    assert Value(logistic_state).read(Step(1, "0.5", "0.8")) == 0.8
    assert Value(logistic_state).name == "x"


def test_a_state_no_logistic_orbit_could_hold_is_refused_by_the_observable():
    # The same refusal the map makes when it writes one, because it is the same function.
    with pytest.raises(ObserveError, match="step 2: a logistic state is a number in 0..1"):
        readings((Value(logistic_state),), Step(2, "0.5", "1.5"))


def test_the_logistic_map_costs_no_checkpoint(tmp_path):
    # Pure arithmetic, so nothing here should reach for torch: the model map's import is inside
    # the function that builds it, and a run of this map must never call that function.
    code = (
        "import sys; from pathlib import Path; from uni.cli import main;"
        "main(['loop','--map','logistic','--start','0.5','--steps','3','--value','3.2'], {}, Path.cwd());"
        "print('torch' in sys.modules)"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=tmp_path)
    assert run.stdout.split()[-1] == "False"


def test_the_flags_of_the_other_map_are_refused_rather_than_dropped(capsys):
    # Carried over from a model run and ignored without a word, these would read back off the
    # command line as settings this orbit had honoured.
    for flag, value in (("--template", "rewrite"), ("--knob", "formality")):
        argv = ["loop", "--map", "logistic", "--start", "0.5", "--steps", "1", "--value", "3.2", flag, value]
        assert command(argv) == EXIT_CONFIG
        assert f"{flag} does not describe the logistic map" in capsys.readouterr().err


@pytest.mark.parametrize("flag, value", [("--text", "a"), ("--layer", "23")])
def test_the_model_map_refuses_the_response_maps_flags(capsys, flag, value):
    assert command(["loop", "--template", "rewrite", "--start", "x", "--steps", "1", flag, value]) == EXIT_CONFIG
    assert f"{flag} does not describe the model map, which reads --template, --knob" in capsys.readouterr().err


def test_r_is_asked_for_rather_than_defaulted(capsys):
    # 0 is a legal r, so a forgotten --value would run the map that sends every state to zero and
    # be answered `period 1` - a period claim, this project's whole output, about nobody's choice.
    assert command(["loop", "--map", "logistic", "--start", "0.5", "--steps", "5"]) == EXIT_CONFIG
    assert "the logistic map's parameter is r; pass --value" in capsys.readouterr().err


@pytest.mark.parametrize("flag, value", [("--template", "rewrite"), ("--knob", "formality"), ("--value", None)])
def test_a_refusal_of_this_map_costs_no_checkpoint(tmp_path, flag, value):
    # The flags are refused by the map, not by argparse, and that is the point: a converter on
    # --knob would read and checksum a direction file, and load torch to hold it, before the run
    # it belongs to had been refused. Every refusal here has to stay cheaper than the run.
    argv = ["loop", "--map", "logistic", "--start", "0.5", "--steps", "1", "--value", "3.2"]
    argv = argv[:-2] if value is None else [*argv, flag, value]
    code = f"import sys; from pathlib import Path; from uni.cli import main;\nprint(main({argv!r}, {{}}, Path.cwd()), 'torch' in sys.modules)"
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=tmp_path)
    assert run.stdout.split() == [str(EXIT_CONFIG), "False"]


def test_every_map_flag_is_one_this_map_refuses(capsys):
    # Derived from the parser the map flags are declared in, not copied from it: a flag added
    # there and nowhere else is refused here rather than accepted and silently dropped.
    assert MAP_OPTIONS == ("template", "knob", "text", "layer")
    for flag in MAP_OPTIONS:
        argv = ["loop", "--map", "logistic", "--start", "0.5", "--steps", "1", "--value", "3.2", f"--{flag}", "1"]
        assert command(argv) == EXIT_CONFIG
        assert f"--{flag} does not describe the logistic map, which reads no flag" in capsys.readouterr().err


def test_an_r_that_is_a_whole_number_writes_a_file_that_reads_back(tmp_path):
    # `Logistic(3)` is a legal map, and an int r would be written as `3`, which the parser reads
    # back as an int and refuses - and which `name` would hash into a second file for one orbit.
    written = Trajectory(Logistic(3).spec, Logistic(3).value, "0.5", ("0.375",))
    assert written.value == 0.375 * 8  # 3.0, held as the float the file has to carry
    assert read_trajectory(write_trajectory(written, tmp_path)) == written
    assert written.name == Trajectory(Logistic(3.0).spec, 3.0, "0.5", ("0.375",)).name


def test_the_model_map_still_needs_its_template(capsys, monkeypatch):
    monkeypatch.setattr("uni.pinned.load_pinned", lambda: pytest.fail("the checkpoint was read before the flags were"))
    assert command(["loop", "--start", "x", "--steps", "1"]) == EXIT_CONFIG
    assert "pass --template" in capsys.readouterr().err
