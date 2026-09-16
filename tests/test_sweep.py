"""The sweep runner, on the map whose cells cost nothing to run.

The property under test is resumption, and it is a property of the files rather than of a record
kept beside them: a cell is done when its trajectory is on disk, so these tests delete cells and
check what a rerun does about them.
"""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from uni.cli import EXIT_CONFIG, Kind, logistic_value, main
from uni.loop import read_trajectory, trajectory_name
from uni.maps import LOGISTIC, Logistic
from uni.sweep import MANIFEST, Sweep, SweepError, grid, pending, read_sweep, write_sweep


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


def test_a_status_query_leaves_nothing_behind(tmp_path, monkeypatch, capsys):
    # A question that created a directory would be an answer that changed what it had just
    # measured: every mistyped or exploratory query would leave an empty sweep behind, and nothing
    # afterwards could tell one of those from a sweep that was started and abandoned.
    monkeypatch.setattr("uni.cli.SWEEPS", tmp_path)
    argv = ["sweep", "--map", "logistic", "--grid", "3.2:3.5:4", "--start", "0.5", "--steps", "6", "--status"]
    assert command(argv) == 0
    assert "0 of 4 cells done, 4 to run" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_naming_a_model_sweep_costs_neither_torch_nor_a_checkpoint(tmp_path):
    # `--status` says how many cells are done. A family holds the pinned configuration rather than
    # a loaded model, so answering that reads pinned.toml and the directory and nothing else - and
    # on a machine with no Metal, `Model.__init__` would not have been a slow answer but a
    # RuntimeError traceback, which is to say no answer at all.
    argv = ["sweep", "--template", "rewrite", "--grid", "0:0:1", "--start", "x", "--steps", "1", "--status"]
    code = f"import sys; from pathlib import Path; from uni.cli import main;\nprint(main({argv!r}, {{}}, Path.cwd()), 'torch' in sys.modules)"
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=tmp_path)
    assert run.stdout.split()[-2:] == ["0", "False"]
    assert list(tmp_path.iterdir()) == []


def test_a_start_given_twice_is_refused(capsys, tmp_path, monkeypatch):
    # One value and one start are one cell and one file. Run twice, the second run overwrites the
    # first for no new data, and the total printed is one the directory can never hold.
    monkeypatch.setattr("uni.cli.SWEEPS", tmp_path)
    argv = ["sweep", "--map", "logistic", "--grid", "3.2:3.5:4", "--steps", "6", "--start", "0.5", "--start", "0.5"]
    assert command(argv) == EXIT_CONFIG
    assert "starts names '0.5' twice" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("text", ["3.5:3.5:5", "3.5:3.5:200"])
def test_a_grid_that_names_one_value_many_times_is_refused(text):
    # The mirror of the rule below it: a grid names one value exactly when its ends are equal.
    # Accepted, this is N runs of one cell, each overwriting the last.
    with pytest.raises(SweepError, match="holds one value, not"):
        grid(text)


@pytest.mark.parametrize("text", ["nan:4:3", "0:1e999:2", "-1e999:1:2", "nan:nan:1"])
def test_a_grid_whose_ends_are_not_numbers_is_refused(text):
    # `--value` goes through `finite` for this reason and a grid is the same quantity: nan and the
    # infinities are not JSON, so a manifest naming one is a file only Python reads back. nan is
    # refused before the ends are compared, or `nan:nan:1` - whose ends are equal in every sense
    # but ==  - would be answered "write nan:nan:1, not 'nan:nan:1'".
    # (Spelled 1e999 rather than as a word: the run host's name is a short common word, and
    # tests/test_no_identity.py fails on any tracked file that carries one.)
    with pytest.raises(SweepError, match="between two finite numbers"):
        grid(text)


def test_a_grid_the_map_refuses_is_refused_whole_and_before_anything_is_written(capsys, tmp_path, monkeypatch):
    # Every value is offered to the map up front, as every value is offered to the knob. Otherwise
    # this writes its manifest, runs the cells below r = 4, and dies on the first one above it -
    # and every resume marches to the same wall again, against a total it can never reach.
    monkeypatch.setattr("uni.cli.SWEEPS", tmp_path)
    argv = ["sweep", "--map", "logistic", "--grid", "3.9:4.2:5", "--start", "0.5", "--steps", "6"]
    assert command(argv) == EXIT_CONFIG
    assert "r must be in 0..4" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


class Drifting:
    """A family whose maps are not at the value they were asked for."""

    @property
    def spec(self):
        return dict(LOGISTIC)

    def holds(self, states):
        return None

    def at(self, value):
        return Logistic(value / 2)


def test_a_cell_that_would_write_a_file_this_sweep_cannot_find_stops_it(capsys, tmp_path, monkeypatch):
    # The sweep names each cell's file before running it, and resumption is that name coming true.
    # A map at a value other than the one asked for writes somewhere else, so `pending` would
    # never stop naming the cell and every rerun would run it again, for ever.
    monkeypatch.setattr("uni.cli.SWEEPS", tmp_path)
    monkeypatch.setitem(main.__globals__["MAPS"], "drifting", Kind(lambda args, values: Drifting(), logistic_value))
    argv = ["sweep", "--map", "drifting", "--grid", "3.2:3.5:4", "--start", "0.5", "--steps", "6"]
    assert command(argv) == EXIT_CONFIG
    assert "does not describe itself the same way twice" in capsys.readouterr().err
    (home,) = tmp_path.iterdir()
    assert cells(home) == []  # it stopped at the first cell rather than filling the directory


@pytest.mark.parametrize(
    "raw, message",
    [
        ('{"map": {}, "values": [3.2, 3.2], "starts": ["a"], "steps": 2}', "values names 3.2 twice"),
        ('{"map": {}, "values": [3.2], "starts": ["a", "a"], "steps": 2}', "starts names 'a' twice"),
        ('{"map": {}, "values": [Infinity], "starts": ["a"], "steps": 2}', "values are finite numbers"),
        ('{"map": {}, "values": [], "starts": ["a"], "steps": 2}', "at least one of values"),
        ('{"map": {}, "values": [3.2], "starts": [], "steps": 2}', "at least one of starts"),
        ('{"map": {}, "values": [3.2], "starts": ["a"], "steps": 0}', "steps each cell at least once"),
    ],
)
def test_a_manifest_that_does_not_describe_a_sweep_is_refused(tmp_path, raw, message):
    # The invariant belongs to the sweep and not to the grid: a repeated `--start` reaches it
    # without passing through one, and `Infinity` is a value only Python's own reader hands back.
    path = tmp_path / MANIFEST
    path.write_text(raw)
    with pytest.raises(SweepError, match=message):
        read_sweep(path)


def test_a_start_the_map_cannot_step_is_refused_before_anything_is_written(capsys, tmp_path, monkeypatch):
    # The starts get the pass the values get. Without it this writes its manifest, runs every cell
    # of the first start, dies on the second - and dies in the same place on every resume, which
    # is the failure the values were checked up front to prevent.
    monkeypatch.setattr("uni.cli.SWEEPS", tmp_path)
    argv = ["sweep", "--map", "logistic", "--grid", "2.8:4.0:200", "--start", "0.5", "--start", "0.10", "--steps", "5"]
    assert command(argv) == EXIT_CONFIG
    assert "write 0.1, not '0.10'" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_two_values_are_one_cell_exactly_when_they_are_one_file():
    # 0.0 and -0.0 are equal and hash alike, so a set counts them as one - but `trajectory_name`
    # hashes the spelling, and they name two files. Refusing them as a repeated value would be a
    # refusal the directory disagrees with, so the check is keyed by the spelling too.
    assert len({cell.name for cell in Sweep(LOGISTIC, (0.0, -0.0), ("0.5",), 4).cells}) == 2
