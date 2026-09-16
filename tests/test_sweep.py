"""The sweep runner, on the map whose cells cost nothing to run.

The property under test is resumption, and it is a property of the files rather than of a record
kept beside them: a cell is done when its trajectory is on disk, so these tests delete cells and
check what a rerun does about them.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from uni.cli import EXIT_CONFIG, main
from uni.loop import read_trajectory, trajectory_name
from uni.sweep import MANIFEST, Sweep, SweepError, grid, pending, read_sweep, write_sweep

LOGISTIC = {"kind": "logistic"}


def command(argv):
    return main(argv, {}, Path.cwd())


def sweep(tmp_path, monkeypatch, gridtext="3.2:3.5:4", starts=("0.5",), steps=6, extra=()):
    monkeypatch.setattr("uni.cli.SWEEPS", tmp_path)
    argv = ["sweep", "--map", "logistic", "--grid", gridtext, "--steps", str(steps)]
    for start in starts:
        argv += ["--start", start]
    assert command([*argv, *extra]) == 0
    (home,) = tmp_path.iterdir()
    return home, argv


def cells(home):
    return sorted(p.name for p in home.glob("*.json") if p.name != MANIFEST)


def test_a_sweep_runs_every_cell_of_the_grid_from_every_start(tmp_path, monkeypatch):
    home, _ = sweep(tmp_path, monkeypatch, starts=("0.5", "0.31"))
    assert len(cells(home)) == 8  # four values, two starts
    assert {read_trajectory(home / name).value for name in cells(home)} == set(grid("3.2:3.5:4"))
    assert {read_trajectory(home / name).start for name in cells(home)} == {"0.5", "0.31"}


def test_the_manifest_says_what_the_sweep_is(tmp_path, monkeypatch):
    home, _ = sweep(tmp_path, monkeypatch)
    assert read_sweep(home / MANIFEST) == Sweep(LOGISTIC, grid("3.2:3.5:4"), ("0.5",), 6)


def test_the_manifest_records_no_progress_of_its_own(tmp_path, monkeypatch):
    # [LAW:one-source-of-truth] the trajectories are the record of what has been done. A count
    # kept beside them is a second clock, and a run killed between writing a cell and updating it
    # would leave the two disagreeing about work that is plainly on disk.
    home, _ = sweep(tmp_path, monkeypatch)
    recorded = json.loads((home / MANIFEST).read_bytes())
    assert sorted(recorded) == ["map", "starts", "steps", "values"]


def test_rerunning_a_finished_sweep_recomputes_nothing(tmp_path, monkeypatch):
    home, argv = sweep(tmp_path, monkeypatch)
    before = {p.name: p.stat().st_mtime_ns for p in home.iterdir()}
    assert command(argv) == 0
    assert {p.name: p.stat().st_mtime_ns for p in home.iterdir()} == before


def test_a_sweep_interrupted_partway_finishes_the_cells_it_is_missing(tmp_path, monkeypatch, capsys):
    home, argv = sweep(tmp_path, monkeypatch)
    lost = cells(home)[1]
    kept = {p.name: p.stat().st_mtime_ns for p in home.iterdir() if p.name != lost}
    (home / lost).unlink()

    capsys.readouterr()
    assert command(argv) == 0
    printed = capsys.readouterr().out
    assert "3 of 4 cells done, 1 to run" in printed
    assert lost in printed
    # The cells that were already there are not rewritten, which is the whole of resumability:
    # an overnight sweep that loses its ssh session picks up where it stopped.
    assert {p.name: p.stat().st_mtime_ns for p in home.iterdir() if p.name != lost} == kept
    assert len(cells(home)) == 4


def test_status_says_how_far_it_got_and_runs_nothing(tmp_path, monkeypatch, capsys):
    home, argv = sweep(tmp_path, monkeypatch)
    for name in cells(home)[2:]:
        (home / name).unlink()
    capsys.readouterr()
    assert command([*argv, "--status"]) == 0
    assert "2 of 4 cells done, 2 to run" in capsys.readouterr().out
    assert len(cells(home)) == 2  # it said so and did nothing about it


def test_a_grid_of_one_value_runs(tmp_path, monkeypatch):
    home, _ = sweep(tmp_path, monkeypatch, gridtext="3.2:3.2:1")
    assert len(cells(home)) == 1
    assert read_trajectory(home / cells(home)[0]).value == 3.2


def test_a_grid_that_ends_on_the_edge_of_a_map_sweeps_to_it(tmp_path, monkeypatch):
    # r = 4 is legal and r = 4.0000000000000004 is not, so this is the sweep that would have
    # died at its last cell had the grid only landed near where it was told to end.
    home, _ = sweep(tmp_path, monkeypatch, gridtext="3.9:4.0:5", steps=4)
    assert max(read_trajectory(home / name).value for name in cells(home)) == 4.0
    assert len(cells(home)) == 5


def test_a_cell_is_named_before_it_is_run(tmp_path, monkeypatch):
    # The name is a hash of what fixes the orbit, all of which the sweep knows in advance. That
    # is what lets "is this cell done" be a question asked of the files themselves.
    home, _ = sweep(tmp_path, monkeypatch, gridtext="3.2:3.2:1", steps=6)
    assert cells(home) == [trajectory_name(LOGISTIC, 3.2, "0.5", 6)]


def test_the_same_sweep_resumes_its_own_directory(tmp_path, monkeypatch):
    first, _ = sweep(tmp_path, monkeypatch)
    second, _ = sweep(tmp_path, monkeypatch)
    assert first == second


@pytest.mark.parametrize(
    "change",
    [{"map": {"kind": "other"}}, {"values": (3.2, 3.6)}, {"starts": ("0.4",)}, {"steps": 7}],
    ids=["map", "values", "starts", "steps"],
)
def test_a_different_sweep_keeps_its_own_directory(change):
    # Everything that fixes the sweep is in the name, so no two sweeps resume each other's work.
    base = Sweep(LOGISTIC, (3.2, 3.5), ("0.5",), 6)
    assert replace(base, **change).name != base.name


def test_the_grid_includes_the_value_it_ends_at():
    assert grid("0:1:3") == (0.0, 0.5, 1.0)
    assert grid("2.8:4.0:5") == pytest.approx((2.8, 3.1, 3.4, 3.7, 4.0))


@pytest.mark.parametrize("count", [2, 5, 17, 199, 200, 335])
def test_a_grid_ends_exactly_where_it_was_told_to(count):
    # Not a rounding nicety. `first + step * (count - 1)` misses `last` for about one grid in
    # twenty, and a logistic sweep to r = 4 - the edge of the map's range - would then end on an
    # r of 4.0000000000000004, which Logistic refuses: the run dies at its last cell.
    values = grid(f"2.8:4.0:{count}")
    assert (values[0], values[-1]) == (2.8, 4.0)
    assert len(values) == count
    assert values == tuple(sorted(values))


@pytest.mark.parametrize(
    "text, message",
    [
        ("2.8:4.0", "a grid is FROM:TO:COUNT"),
        ("2.8:4.0:2:1", "a grid is FROM:TO:COUNT"),
        ("a:b:3", "two numbers and a whole count"),
        ("2.8:4.0:2.5", "two numbers and a whole count"),
        ("2.8:4.0:0", "at least one value"),
        ("2.8:4.0:1", "a grid of one value is a single value"),
    ],
)
def test_a_grid_that_is_not_one_is_refused(text, message):
    with pytest.raises(SweepError, match=message):
        grid(text)


def test_a_grid_the_command_cannot_read_is_reported_rather_than_raised(capsys, tmp_path, monkeypatch):
    # argparse catches only its own error type, so this is read by the command and not by a
    # converter: a typo is the user's to fix and reaches them as `uni: ...`, not as a traceback.
    monkeypatch.setattr("uni.cli.SWEEPS", tmp_path)
    argv = ["sweep", "--map", "logistic", "--grid", "2.8:4.0", "--start", "0.5", "--steps", "5"]
    assert command(argv) == EXIT_CONFIG
    assert "uni: a grid is FROM:TO:COUNT" in capsys.readouterr().err


def test_the_flags_of_the_model_map_are_refused_by_a_logistic_sweep(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr("uni.cli.SWEEPS", tmp_path)
    argv = ["sweep", "--map", "logistic", "--grid", "3.2:3.2:1", "--start", "0.5", "--steps", "5", "--knob", "formality"]
    assert command(argv) == EXIT_CONFIG
    assert "--knob describes the model map" in capsys.readouterr().err


def test_a_grid_with_no_knob_to_turn_is_refused_before_the_checkpoint(capsys, tmp_path, monkeypatch):
    # A sweep is hundreds of model runs. The whole grid is offered to the knob up front, so a
    # sweep that could never have steered anything is refused in milliseconds rather than after
    # a checkpoint load and a first cell.
    monkeypatch.setattr("uni.cli.SWEEPS", tmp_path)
    monkeypatch.setattr("uni.pinned.load_pinned", lambda: pytest.fail("the checkpoint was read before the grid was refused"))
    argv = ["sweep", "--template", "rewrite", "--grid", "0:2:5", "--start", "x", "--steps", "1"]
    assert command(argv) == EXIT_CONFIG
    assert "uni: there is no knob to turn" in capsys.readouterr().err


def test_a_manifest_that_is_not_one_is_refused(tmp_path):
    path = tmp_path / MANIFEST
    for text, message in (
        ('{"values": [', "is not JSON"),
        ("[]", "must hold a JSON object"),
        ('{"map": {}, "values": [1.0], "starts": ["a"]}', "steps is missing"),
        ('{"map": {}, "values": [1], "starts": ["a"], "steps": 2}', "values must all be numbers"),
        ('{"map": {}, "values": [1.0], "starts": [2], "steps": 2}', "starts must all be strings"),
    ):
        path.write_text(text)
        with pytest.raises(SweepError, match=message):
            read_sweep(path)


def test_a_manifest_that_is_not_there_is_refused_rather_than_raised(tmp_path):
    with pytest.raises(SweepError, match="cannot be read: No such file"):
        read_sweep(tmp_path / "nope.json")


def test_a_sweep_round_trips_through_its_manifest(tmp_path):
    original = Sweep(LOGISTIC, (3.2, 3.5), ("a", "héllo\n"), 4)
    assert read_sweep(write_sweep(original, tmp_path) / MANIFEST) == original


def test_pending_is_every_cell_with_no_trajectory_on_disk(tmp_path):
    original = Sweep(LOGISTIC, (3.2, 3.5), ("0.5",), 4)
    home = write_sweep(original, tmp_path)
    assert len(pending(original, home)) == 2
    (home / next(iter(original.cells)).name).write_text("{}")
    assert len(pending(original, home)) == 1
