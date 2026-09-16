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
from uni.observe import ObserveError, observables
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


def test_the_burn_in_drops_that_many_steps_from_the_front(tmp_path, monkeypatch):
    written, home = sweep(tmp_path, monkeypatch, steps=8)
    whole = read(written, home, "x", burn_in=0)[0]
    settled = read(written, home, "x", burn_in=5)[0]
    assert settled.numbers == whole.numbers[5:]


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
        named(observables(trajectory), "logprob")


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
    assert sorted(p.name.split("-", 1)[1] for p in out.glob("*.png")) == ["x-orbit.png", "x-return.png"]
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
    # An empty figure is a file that looks like an answer. The count in the message is what tells
    # a burn-in that ate everything apart from a sweep that has not run yet.
    _, home = sweep(tmp_path, monkeypatch, steps=8)
    out = tmp_path / "figures"
    assert command(["plot", str(home), "--observable", "x", "--burn-in", "99", "--out", str(out)]) == EXIT_CONFIG
    assert "holds no readings to draw" in capsys.readouterr().err
    assert not out.exists()  # and writes nothing on the way to saying so


def test_a_directory_that_holds_no_sweep_is_refused(tmp_path, monkeypatch, capsys):
    assert command(["plot", str(tmp_path), "--observable", "x"]) == EXIT_CONFIG
    assert "cannot be read: No such file" in capsys.readouterr().err
