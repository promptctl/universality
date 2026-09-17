"""The two pictures, on the map whose cells cost nothing to run.

What is under test is the arithmetic between a sweep directory and a set of points: which cells
are read, which steps survive the burn-in, and what each picture does with the numbers. Whether a
dot lands where matplotlib puts it is not this file's business.
"""

from pathlib import Path

import pytest

from uni.cli import EXIT_CONFIG, main
from uni.figure import Readings, named, orbit_diagram, read, return_map
from uni.loop import read_trajectory
from uni.observe import ObserveError, Weights, observables, steps
from uni.sweep import MANIFEST, finished, pending, read_sweep


def command(argv):
    return main(argv, {}, Path.cwd())


def sweep(tmp_path, monkeypatch, gridtext="3.2:3.5:4", starts=("0.2",), steps=8):
    monkeypatch.setattr("uni.cli.SWEEPS", tmp_path)
    argv = ["sweep", "--map", "logistic", "--grid", gridtext, "--steps", str(steps)]
    for start in starts:
        argv += ["--start", start]
    assert command(argv) == 0
    (home,) = tmp_path.iterdir()
    return read_sweep(home / MANIFEST), home


def test_every_finished_cell_is_read_once_in_sweep_order(tmp_path, monkeypatch):
    written, home = sweep(tmp_path, monkeypatch)
    series = read(written, home, "x", burn_in=0)
    assert [one.value for one in series] == list(written.values)
    assert all(len(one.numbers) == written.steps for one in series)


def test_a_sweep_still_running_is_drawn_from_what_is_on_disk(tmp_path, monkeypatch):
    # A sweep is normally incomplete while it runs, and a picture of what it has so far is the
    # point of persisting cells one at a time. `finished` and `pending` are the two halves of one
    # predicate, so what a picture draws and what a rerun would run cannot overlap.
    written, home = sweep(tmp_path, monkeypatch)
    lost = finished(written, home)[1]
    (home / lost.name).unlink()
    series = read(written, home, "x", burn_in=0)
    assert len(series) == len(written.values) - 1
    assert lost in pending(written, home)


def test_the_burn_in_keeps_the_steps_uni_observe_would_keep(tmp_path, monkeypatch):
    # One meaning for one flag. `detect` passes over a sequence whose element 0 is the start, so
    # its --burn-in N keeps step N onward; a slice of `steps()` counts from step 1 and would keep
    # step N + 1. Paired on one sweep - as EXPERIMENTS.md pairs them - that is a period measured
    # over different states than the picture beside it is drawn from.
    written, home = sweep(tmp_path, monkeypatch, steps=8)
    trajectory = read_trajectory(home / finished(written, home)[0].name)
    orbit = (trajectory.start, *trajectory.states)
    for burn_in in (0, 1, 2, 5):
        drawn = [step.state for step in steps(trajectory) if step.index >= burn_in]
        assert len(read(written, home, "x", burn_in)[0].numbers) == len(drawn)
        examined = list(orbit[burn_in:])  # what `uni observe --burn-in` hands the detector
        # The same states, but for the start: it is step 0, and no observable reads it, so a
        # picture has nothing to plot for it. Past step 0 the two commands keep the same orbit.
        assert drawn == (examined[1:] if burn_in == 0 else examined)


def test_a_burn_in_past_the_end_leaves_a_cell_with_nothing(tmp_path, monkeypatch):
    # Not an error here: it is the whole sweep having nothing left that is worth refusing, and
    # that is the command's call to make, once, rather than this function's per cell.
    written, home = sweep(tmp_path, monkeypatch, steps=8)
    assert all(one.numbers == () for one in read(written, home, "x", burn_in=99))


def test_a_scratch_file_in_the_directory_is_never_read_as_a_cell(tmp_path, monkeypatch):
    # A signal that ends a run outright leaves its scratch file behind. The cells are regenerated
    # from the manifest rather than globbed, so litter cannot become a data point.
    written, home = sweep(tmp_path, monkeypatch)
    (home / "9999999999999999.json.4242.partial").write_bytes(b"not a trajectory")
    assert len(read(written, home, "x", burn_in=0)) == len(written.values)


def test_an_observable_this_sweep_does_not_read_is_refused_by_name(tmp_path, monkeypatch):
    written, home = sweep(tmp_path, monkeypatch)
    trajectory = read_trajectory(home / finished(written, home)[0].name)
    with pytest.raises(ObserveError, match="no observable 'logprob' for this sweep; it reads length, x"):
        named(observables(trajectory, Weights()), "logprob")


def test_the_return_map_pairs_each_reading_with_the_one_after_it():
    series = (Readings(3.5, (0.1, 0.2, 0.3)), Readings(3.9, (0.4, 0.5)))
    picture = return_map(series, "x")
    assert [(p.x, p.y, p.shade) for p in picture.points] == [
        (0.1, 0.2, 3.5), (0.2, 0.3, 3.5), (0.4, 0.5, 3.9),
    ]
    assert picture.shade_label == "value"  # a grid is a fan of colours; one value is one colour


def test_a_cell_with_one_reading_has_no_pair_and_so_no_point():
    # A step and the step after it are what a return map is, so a cell that cannot show one
    # contributes nothing rather than a dot on the diagonal it never visited.
    assert return_map((Readings(3.5, (0.1,)),), "x").points == ()


def test_the_orbit_diagram_puts_every_reading_over_the_value_it_was_taken_at():
    series = (Readings(3.5, (0.1, 0.2)), Readings(3.9, (0.4,)))
    picture = orbit_diagram(series, "x")
    assert [(p.x, p.y) for p in picture.points] == [(3.5, 0.1), (3.5, 0.2), (3.9, 0.4)]
    # One shade, and no key: the value is already the x axis, so colouring by it says it twice.
    assert {p.shade for p in picture.points} == {0.0}
    assert picture.shade_label is None


def test_a_sweep_of_one_value_renders_a_single_column(tmp_path, monkeypatch, capsys):
    # The epic's checkpoint names this edge case: it must render, not error.
    written, home = sweep(tmp_path, monkeypatch, gridtext="3.9:3.9:1", steps=40)
    out = tmp_path / "figures"
    assert command(["plot", str(home), "--observable", "x", "--burn-in", "20", "--out", str(out)]) == 0
    assert sorted(p.name.split("-", 1)[1] for p in out.glob("*.png")) == ["x-burn20-orbit.png", "x-burn20-return.png"]
    column = orbit_diagram(read(written, home, "x", burn_in=20), "x")
    assert {point.x for point in column.points} == {3.9}


def test_a_sweep_of_fixed_points_renders_a_line(tmp_path, monkeypatch):
    # The other checkpoint edge case. Below r = 3 every orbit settles on 1 - 1/r, so each cell
    # contributes one y repeated: a line, and a return map that is a run of points on the diagonal.
    written, home = sweep(tmp_path, monkeypatch, gridtext="2.5:2.9:5", steps=400)
    series = read(written, home, "x", burn_in=300)
    # Settled on the analytic fixed point, not merely on a float that stopped moving: at r = 2.9
    # the approach is 0.9 per step, so how long it takes to get there is itself worth pinning.
    assert all(max(one.numbers) - min(one.numbers) < 1e-9 for one in series)
    assert all(one.numbers[-1] == pytest.approx(1 - 1 / one.value) for one in series)
    # And a return map of a fixed point is points on the diagonal: x maps to itself.
    assert all(point.x == pytest.approx(point.y) for point in return_map(series, "x").points)
    out = tmp_path / "figures"
    assert command(["plot", str(home), "--observable", "x", "--burn-in", "300", "--out", str(out)]) == 0
    assert len(list(out.glob("*.png"))) == 2


def test_a_sweep_with_nothing_settled_is_refused_rather_than_drawn_empty(tmp_path, monkeypatch, capsys):
    # An empty figure is a file that looks like an answer. The counts in the message are what tell
    # a burn-in that ate everything apart from a sweep that has not run yet.
    _, home = sweep(tmp_path, monkeypatch, steps=8)
    out = tmp_path / "figures"
    assert command(["plot", str(home), "--observable", "x", "--burn-in", "99", "--out", str(out)]) == EXIT_CONFIG
    printed = capsys.readouterr().err
    assert "nothing to draw the return and orbit map of" in printed
    assert "4 of 4 cells are on disk" in printed and "leaves 0 readings" in printed
    assert not out.exists()  # and writes nothing on the way to saying so


def test_a_burn_in_that_leaves_one_step_refuses_both_rather_than_drawing_a_blank_return_map(tmp_path, monkeypatch, capsys):
    # The return map is the strict half: it pairs each reading with the one after it, so a cell
    # that can show only one reading has no pair. The orbit diagram of the same sweep is fine,
    # which is exactly how a blank figure gets written beside a good one and exits 0.
    _, home = sweep(tmp_path, monkeypatch, steps=6)
    out = tmp_path / "figures"
    assert command(["plot", str(home), "--observable", "x", "--burn-in", "6", "--out", str(out)]) == EXIT_CONFIG
    printed = capsys.readouterr().err
    assert "nothing to draw the return map of" in printed
    assert "a return map needs two from one cell, an orbit diagram one" in printed
    assert not out.exists()  # neither picture is written when only one of them can be


def test_a_directory_that_holds_no_sweep_is_refused(tmp_path, monkeypatch, capsys):
    assert command(["plot", str(tmp_path), "--observable", "x"]) == EXIT_CONFIG
    assert "cannot be read: No such file" in capsys.readouterr().err


def model_sweep(tmp_path):
    """A model sweep on disk, whose cells no model has to exist to have written."""
    from uni.loop import Trajectory, write_trajectory
    from uni.maps import model_spec
    from uni.pinned import load_pinned
    from uni.sweep import Sweep
    from uni.template import load_templates

    spec = model_spec(load_pinned(), load_templates()["rewrite"], None)
    values, start = (0.0, 1.0, 2.0), "hello"
    written = Sweep(spec, values, (start,), 2)
    home = written.home(tmp_path)
    for value in values:
        write_trajectory(Trajectory(spec, value, start, ("a", "bb")), home)
    return written, home


def test_a_picture_of_what_every_map_answers_reads_no_checkpoint_at_all(tmp_path, monkeypatch):
    # Saying what a model orbit can be read for must not cost the reading. The length of a state
    # is characters in JSON already on disk, so drawing it loads nothing - which is also the
    # difference, on a machine with no Metal, between an answer and an uncaught RuntimeError.
    loads = []
    monkeypatch.setattr("uni.observe.Model", lambda pinned: loads.append(pinned))
    written, home = model_sweep(tmp_path)
    series = read(written, home, "length", burn_in=0)
    assert [one.numbers for one in series] == [(1.0, 2.0)] * 3  # the lengths of "a" and "bb"
    assert loads == []


def test_a_picture_of_a_model_observable_reads_the_checkpoint_once(tmp_path, monkeypatch):
    # And when a reading does need the model, every cell of the sweep is read through one: the
    # sweep's own description says they are all the same model, so loading per cell is waste.
    class Stub:
        def reply_logprob(self, prompt, state, additions):
            return -1.5

    loads = []
    monkeypatch.setattr("uni.observe.Model", lambda pinned: (loads.append(pinned), Stub())[1])
    written, home = model_sweep(tmp_path)
    series = read(written, home, "logprob", burn_in=0)
    assert [one.numbers for one in series] == [(-1.5, -1.5)] * 3
    assert len(loads) == 1


def test_a_figure_is_named_by_the_sweep_and_not_by_the_directory_it_was_found_in(tmp_path, monkeypatch):
    # They agree for a sweep this program wrote. When they do not - a copied directory, a renamed
    # one - it is the manifest that says which sweep this is a picture of.
    written, home = sweep(tmp_path, monkeypatch)
    copied = home.parent / "a-name-of-my-own"
    copied.mkdir()
    for path in home.iterdir():
        copied.joinpath(path.name).write_bytes(path.read_bytes())
    out = tmp_path / "figures"
    assert command(["plot", str(copied), "--observable", "x", "--out", str(out)]) == 0
    assert sorted(p.name for p in out.glob("*.png")) == [f"{written.name}-x-burn0-orbit.png", f"{written.name}-x-burn0-return.png"]


def test_a_cell_that_cannot_be_read_says_which_cell(tmp_path, monkeypatch, capsys):
    # The messages underneath are about one orbit and were written when the caller had named it.
    # Over a sweep the cell is the file to go and look at, and it arrives after every cell before
    # it has already been read.
    written, home = sweep(tmp_path, monkeypatch)
    broken = finished(written, home)[1]
    # A kind this build cannot read, which `observables` refuses before any reading is taken - the
    # half of the per-cell work the message used to come out of unattributed.
    for raw, message in (
        # The three families that refuse a cell, which are three sibling error types: the file's
        # own shape, what its states are made of, and - for a steered orbit - the direction file.
        ('{"map": {"kind": "logistic"}, "value": 3.3, "start": "0.2", "states": [1, 2]}', "states must all be strings"),
        ('{"map": {"kind": "henon"}, "value": 3.3, "start": "0.2", "states": ["0.5"]}', "not one this build can read"),
        ('{"map": {"kind": "logistic"}, "value": 3.3, "start": "0.2", "states": ["nope"]}', "a logistic state is a number in 0..1"),
    ):
        (home / broken.name).write_text(raw)
        assert command(["plot", str(home), "--observable", "x", "--out", str(tmp_path / "figures")]) == EXIT_CONFIG
        printed = capsys.readouterr().err
        assert broken.name in printed and message in printed


def test_plot_is_refused_on_the_host_rather_than_drawing_where_nothing_can_reach_it(capsys):
    # The sync has one leg and figures/ is not on it, so this would draw on the host, print a path
    # that does not exist here, and exit 0 as though it had answered.
    assert main(["plot", "--remote", "sweeps/whatever", "--observable", "x"], {}, Path.cwd()) == EXIT_CONFIG
    assert "runs here, not on the host" in capsys.readouterr().err


def test_the_commands_that_stay_here_are_commands_and_the_rest_travel():
    # HERE is a set of names consulted without parsing, because parsing runs the converters and one
    # of them imports torch. What keeps that set in step with the parser is this test, so it asks
    # the parser for its subcommands rather than restating them: a renamed command, or a typo in
    # HERE, is a guard that silently stops guarding, and the failure it stops guarding against is
    # the one drawing a figure onto the host that nothing can reach.
    from uni.cli import HERE, build_parser, stays_here

    # argparse offers no public way to enumerate subcommands; `choices` on the action it made for
    # them is the nearest thing, and a test is the right place to reach for it.
    commands = next(action for action in build_parser()._actions if action.dest == "command").choices
    assert HERE.keys() <= set(commands), "a name in HERE that no command answers to guards nothing"
    assert {name for name in commands if stays_here([name])} == set(HERE)
    assert stays_here(["plot", "sweeps/x", "--observable", "x"])  # and with its arguments, as typed
