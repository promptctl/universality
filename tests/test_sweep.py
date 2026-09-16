"""The sweep runner, on the map whose cells cost nothing to run.

The property under test is resumption, and it is a property of the files rather than of a record
kept beside them: a cell is done when its trajectory is on disk, so these tests delete cells and
check what a rerun does about them.
"""

import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from uni.cli import EXIT_CONFIG, EXIT_INCOMPLETE, Kind, logistic_value, main
from uni.loop import read_trajectory, trajectory_name
from uni.maps import LOGISTIC, Logistic, MapError
from uni.sweep import MANIFEST, Failed, Sweep, SweepError, grid, pending, read_sweep, run_cell, write_sweep


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


def test_a_flag_of_another_map_is_refused_by_a_logistic_sweep(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr("uni.cli.SWEEPS", tmp_path)
    argv = ["sweep", "--map", "logistic", "--grid", "3.2:3.2:1", "--start", "0.5", "--steps", "5", "--knob", "formality"]
    assert command(argv) == EXIT_CONFIG
    assert "--knob does not describe the logistic map" in capsys.readouterr().err


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


class Assorted:
    """A family whose map at each value is whatever the test says, so one sweep meets several.

    Handed factories rather than maps, the way a real family makes one per `at`: `Outgrown` below
    counts its own steps, and a cell run again by a resume has to meet a fresh one.
    """

    def __init__(self, maps=()):
        self.maps = dict(maps)

    @property
    def spec(self):
        return dict(LOGISTIC)

    def holds(self, states):
        return None

    def at(self, value):
        return self.maps.get(value, Logistic)(value)


def drifting(value):
    """A map that is not at the value it was asked for, so it writes a file no cell is looking for."""
    return Logistic(value / 2)


def test_a_cell_that_would_write_a_file_this_sweep_cannot_find_stops_it(capsys, tmp_path, monkeypatch):
    # The sweep names each cell's file before running it, and resumption is that name coming true.
    # A map at a value other than the one asked for writes somewhere else, so `pending` would
    # never stop naming the cell and every rerun would run it again, for ever.
    monkeypatch.setattr("uni.cli.SWEEPS", tmp_path)
    everywhere = {value: drifting for value in grid("3.2:3.5:4")}
    monkeypatch.setitem(main.__globals__["MAPS"], "drifting", Kind(lambda args, values: Assorted(everywhere), logistic_value))
    argv = ["sweep", "--map", "drifting", "--grid", "3.2:3.5:4", "--start", "0.5", "--steps", "6"]
    assert command(argv) == EXIT_CONFIG
    printed = capsys.readouterr()
    assert "does not describe itself the same way twice" in printed.err
    (home,) = tmp_path.iterdir()
    assert cells(home) == []  # it stopped at the first cell rather than filling the directory
    # And stopped there rather than carrying on the way it carries on past a cell the map refused.
    # The two are not the same failure: a map that refuses a cell has said something about that
    # cell, and a map that names its file differently than this sweep named it has said something
    # about every cell - so this one is raised where that one is returned, and the run ends here.
    assert "2/4" not in printed.out


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


def test_status_answers_without_asking_the_map_about_the_starts(tmp_path, monkeypatch, capsys):
    # The deliberate edge of putting `holds` below the --status return: a model family answers
    # about starts with the checkpoint, and counting files on disk must not cost one. So --status
    # reports on a sweep whose starts the map would refuse, and running it is what refuses them.
    monkeypatch.setattr("uni.cli.SWEEPS", tmp_path)
    argv = ["sweep", "--map", "logistic", "--grid", "3.2:3.5:4", "--start", "0.10", "--steps", "5"]
    assert command([*argv, "--status"]) == 0
    assert "0 of 4 cells done, 4 to run" in capsys.readouterr().out
    assert command(argv) == EXIT_CONFIG
    assert "write 0.1, not '0.10'" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_another_run_of_the_same_sweep_writes_the_manifest_under_its_own_scratch_name(tmp_path):
    # The manifest gets what the cells get. Two runs of one sweep start together on a fresh
    # directory, both find no manifest, and under one scratch name the second would truncate the
    # bytes the first was about to rename into place - leaving `sweep.json` a prefix of itself,
    # which `read_sweep` refuses for a sweep whose cells are all fine.
    written = Sweep(LOGISTIC, (3.2, 3.5), ("0.5",), 4)
    home = written.home(tmp_path)
    foreign = home / f"{MANIFEST}.partial"  # a name no run here picks, so no run here may write it
    foreign.parent.mkdir(parents=True, exist_ok=True)
    foreign.write_bytes(b"")
    assert write_sweep(written, tmp_path) == home
    assert read_sweep(home / MANIFEST) == written
    assert foreign.read_bytes() == b""  # left alone: another process's scratch file is not ours


def test_a_sweep_of_whole_numbers_writes_a_manifest_it_reads_back(tmp_path):
    # `trajectory_name` floats the value before hashing it, so a sweep at 3 names the cells a
    # sweep at 3.0 names - and would then describe itself in a manifest saying `"values": [3]`,
    # which its own reader refuses as an int. One spelling, fixed where the sweep is made.
    whole = Sweep(LOGISTIC, (3, 4), ("0.5",), 4)
    assert whole == Sweep(LOGISTIC, (3.0, 4.0), ("0.5",), 4)
    home = write_sweep(whole, tmp_path)
    assert read_sweep(home / MANIFEST) == whole


def test_one_value_written_two_ways_is_refused_as_the_one_cell_it_is():
    # 3 and 3.0 are one number and name one file, so they are one cell named twice - the case the
    # spelling-keyed distinctness check would let through if the spellings were not settled first.
    with pytest.raises(SweepError, match="values names 3.0 twice"):
        Sweep(LOGISTIC, (3, 3.0), ("0.5",), 4)


def test_a_sweep_with_nothing_left_asks_the_map_to_hold_no_start(tmp_path, monkeypatch):
    # `holds` is the one question a model family answers with the checkpoint, so a rerun with no
    # cell to run must not ask it. Rerunning the finished command is how this project sees that a
    # sweep is done, and reading half a billion parameters to reprint a number already in hand is
    # the wrong price for that - off Metal it is not a slow answer but a raise where the answer
    # should have been.
    asked = []

    class Watchful:
        """A family that records what it was asked to hold, the way a model family loads for it."""

        @property
        def spec(self):
            return dict(LOGISTIC)

        def holds(self, states):
            asked.append(tuple(states))

        def at(self, value):
            return Logistic(value)

    monkeypatch.setattr("uni.cli.SWEEPS", tmp_path)
    monkeypatch.setitem(main.__globals__["MAPS"], "watchful", Kind(lambda args, values: Watchful(), logistic_value))
    argv = ["sweep", "--map", "watchful", "--grid", "3.2:3.5:4", "--start", "0.5", "--steps", "6"]
    assert command(argv) == 0
    assert asked == [("0.5",)]
    assert command(argv) == 0
    assert asked == [("0.5",), ()]  # nothing left, so nothing asked


@dataclass
class Outgrown:
    """A map that steps `after` times and then has nowhere to put the next state.

    What a model map does when its states - which are the model's own replies - grow until the
    rendered state leaves no room to generate. Which step that happens at is knowable only by
    running to it, so this is the one failure the checks before the first cell cannot cover.
    """

    r: float
    after: int
    taken: int = 0

    @property
    def value(self):
        return self.r

    @property
    def spec(self):
        return dict(LOGISTIC)

    def step(self, state):
        self.taken += 1
        if self.taken > self.after:
            raise MapError("prompt is 32760 tokens; the context limit of 32768 leaves no room to generate")
        return Logistic(self.r).step(state)


def outgrows(after):
    """A factory for a map that gets `after` states out before it has nowhere to put the next."""
    return lambda value: Outgrown(value, after)


def four_cells(tmp_path, monkeypatch, family):
    """The four-cell logistic sweep, run through a family handed in, and the command that runs it."""
    monkeypatch.setattr("uni.cli.SWEEPS", tmp_path)
    monkeypatch.setitem(main.__globals__["MAPS"], "assorted", Kind(lambda args, values: family, logistic_value))
    return ["sweep", "--map", "assorted", "--grid", "3.2:3.5:4", "--start", "0.5", "--steps", "6"]


def test_a_cell_the_map_cannot_run_leaves_the_cells_after_it_to_run(capsys, tmp_path, monkeypatch):
    # Stopping at this cell would stop at it on every resume, because whatever the map refused it
    # for is a property of the cell and the resume runs the same cell again - so the cells after
    # it would be unreachable for good, which is a worse wall than the one the up-front checks
    # exist to prevent: no amount of rerunning gets past this one.
    values = grid("3.2:3.5:4")
    argv = four_cells(tmp_path, monkeypatch, Assorted({values[1]: outgrows(3)}))
    # A run that did less than it was asked does not exit 0 - and does not say what a mistyped
    # --grid says either, because this sweep ran and most of it is on disk.
    assert command(argv) == EXIT_INCOMPLETE
    assert EXIT_INCOMPLETE != EXIT_CONFIG
    printed = capsys.readouterr()
    (home,) = tmp_path.iterdir()
    assert len(cells(home)) == 3
    refused = trajectory_name(LOGISTIC, values[1], "0.5", 6)
    assert refused not in cells(home)  # nothing is written for a cell with no orbit
    # And yet it is named where a person watching sees it. The value on that line is rounded to
    # fit its column, and rounded it is 3.3 - an orbit in another file - so the name is what says
    # which cell stopped.
    assert refused in printed.out
    assert "3 of 4 cells done, 1 to run" in printed.out
    # Named, which is the whole of what the old message did not do: it said token counts and left
    # the reader to work out which of four hundred cells they were about. The value is written as
    # the shortest text that reads back as itself - the rule `logistic_state` holds a state to -
    # because it is what names the cell, and this one rounds to 3.3, which is a different orbit in
    # a different file, and in the chaotic regime a visibly different one.
    assert values[1] != 3.3
    assert f"value {values[1]!r} from '0.5' has no orbit: prompt is 32760 tokens" in printed.err


def test_a_cell_with_no_orbit_leaves_nothing_behind_in_the_sweep_directory(tmp_path, monkeypatch):
    # [LAW:one-source-of-truth] `written` asks a directory listing whether a cell is done, and
    # `pending` and `finished` are its two halves. A failure recorded here would be a second kind
    # of file in a directory of orbits, which every reader of a sweep - those two, the plot, the
    # rsync that brings one home - would have to learn to tell from the real thing.
    values = grid("3.2:3.5:4")
    argv = four_cells(tmp_path, monkeypatch, Assorted({values[1]: outgrows(3)}))
    assert command(argv) == EXIT_INCOMPLETE
    (home,) = tmp_path.iterdir()
    assert sorted(p.name for p in home.iterdir()) == sorted([MANIFEST, *cells(home)])
    assert {read_trajectory(home / name).value for name in cells(home)} == {3.2, values[2], 3.5}


def test_a_cell_that_could_not_run_is_run_again_by_the_next_run(tmp_path, monkeypatch):
    # The point of keeping the failure out of the directory: the cell is still pending, so the run
    # after the cause is gone picks it up with nothing to delete by hand first. A cell recorded as
    # failed would be a cell marked done by a run that did not do it, and nothing would go back.
    values = grid("3.2:3.5:4")
    argv = four_cells(tmp_path, monkeypatch, Assorted({values[1]: outgrows(3)}))
    assert command(argv) == EXIT_INCOMPLETE
    (home,) = tmp_path.iterdir()
    assert len(cells(home)) == 3
    monkeypatch.setitem(main.__globals__["MAPS"], "assorted", Kind(lambda args, vals: Assorted(), logistic_value))
    assert command(argv) == 0
    assert len(cells(home)) == 4
    assert read_trajectory(home / trajectory_name(LOGISTIC, values[1], "0.5", 6)).value == values[1]


def test_a_cell_refused_at_its_very_first_step_is_still_that_cell_being_refused(tmp_path, monkeypatch):
    # How far the orbit got is not the test, however tempting: residual additions large enough to
    # overflow the model's arithmetic refuse the first token of the cell they are too large in,
    # which is a property of that cell's value. Read as "this is about no cell in particular" it
    # would rebuild, at one end of a steering grid, the very wall this feature removes.
    values = grid("3.2:3.5:4")
    argv = four_cells(tmp_path, monkeypatch, Assorted({values[1]: outgrows(0)}))
    assert command(argv) == EXIT_INCOMPLETE
    (home,) = tmp_path.iterdir()
    assert len(cells(home)) == 3
    assert trajectory_name(LOGISTIC, values[1], "0.5", 6) not in cells(home)


def unmakeable(value):
    """A value the family cannot make a map at, the way a direction that fits no checkpoint is."""
    raise MapError("layer must be in 0..23, got 40")


def test_a_family_that_cannot_make_the_map_stops_the_sweep(capsys, tmp_path, monkeypatch):
    # What a map is made out of is the family's, not the value's: a steering direction's layer and
    # the length of its vector are the same in every cell. So a family that cannot make a map at
    # the first cell cannot make one at any of them, and the sweep is refused whole - before the
    # manifest, as a grid the map refuses is - rather than once per cell for a run that can never
    # write a file. An empty sweep directory left behind here is litter nothing ever fills: the
    # spec that named it is the one that has to change before the run can work.
    argv = four_cells(tmp_path, monkeypatch, Assorted({value: unmakeable for value in grid("3.2:3.5:4")}))
    assert command(argv) == EXIT_CONFIG
    printed = capsys.readouterr()
    assert "1/4" not in printed.out  # no cell ran at all, never mind all four
    assert "uni: layer must be in 0..23, got 40" in printed.err
    assert "has no orbit" not in printed.err  # not dressed up as one cell's failure, because it is not
    assert list(tmp_path.iterdir()) == []


def test_the_cells_refused_so_far_are_named_even_when_something_else_ends_the_run(capsys, tmp_path, monkeypatch):
    # The count and the refusals are about what the run did, not about how it stopped. Said only
    # on the way out of a clean loop, this refusal would survive as one line in the scrollback of
    # a sweep that prints thousands, while the message that ended the run got the last word.
    #
    # A cell is written before the refusal on purpose, so the count on the way out is one the
    # count before the run could not have printed: asserted against "0 of 4" - what this sweep
    # starts at - the whole `finally` could be deleted and this test would not notice.
    values = grid("3.2:3.5:4")
    argv = four_cells(tmp_path, monkeypatch, Assorted({values[1]: outgrows(3), values[2]: drifting}))
    assert command(argv) == EXIT_CONFIG
    printed = capsys.readouterr()
    assert "0 of 4 cells done, 4 to run" in printed.out  # what it was going to do, said before it ran
    assert "1 of 4 cells done, 3 to run" in printed.out  # what it did, said on the way out
    assert f"value {values[1]!r} from '0.5' has no orbit" in printed.err
    assert "does not describe itself the same way twice" in printed.err


class Closed:
    """A stdout that cannot carry anything more, the way `uni sweep ... | head` leaves one."""

    def __init__(self, error):
        self.error = error
        self.listening = True

    def write(self, text):
        if not self.listening:
            raise self.error
        return len(text)

    def flush(self):
        if not self.listening:
            raise self.error


def closing(stdout):
    """A drifting map that stops the reader as it is made, so the pipe goes as the run is ending."""

    def make(value):
        stdout.listening = False
        return drifting(value)

    return make


# A closed pipe is the one that happens, and a full disk under `> file` loses the report the same
# way: what the run has to survive is a stream that cannot carry a message, not one particular
# reason it cannot. Both, so the suppression is pinned to the question and not to the `| head`.
@pytest.mark.parametrize("error", [BrokenPipeError(32, "Broken pipe"), OSError(28, "No space left on device")])
def test_a_stdout_that_cannot_carry_the_report_does_not_replace_what_ended_the_run(capsys, tmp_path, monkeypatch, error):
    # The count is printed from a `finally`, and a `finally` that raises replaces the exception
    # that got it there. Nothing further out can put that back, so with the reader gone the
    # refusal this sweep exists to report would reach the user as a traceback from the reporting
    # instead of as `uni: ...` and EXIT_CONFIG. Saying how far a run got to nobody is not a
    # failure of the run.
    values = grid("3.2:3.5:4")
    stdout = Closed(error)
    argv = four_cells(tmp_path, monkeypatch, Assorted({values[0]: outgrows(3), values[1]: closing(stdout)}))
    monkeypatch.setattr(sys, "stdout", stdout)  # after the sweep is described: this run has a reader until cell 2
    assert command(argv) == EXIT_CONFIG
    printed = capsys.readouterr()
    assert "does not describe itself the same way twice" in printed.err  # what ended the run, not what the finally hit
    assert f"value {values[0]!r} from '0.5' has no orbit" in printed.err  # and the refusal it was carrying


def test_a_map_that_refuses_a_cell_is_answered_with_a_value_and_not_an_exception():
    # The decision this ticket settled, pinned where it is made rather than only where it shows:
    # the ways a cell can fail to produce a trajectory differ in what they are about, so they
    # differ in kind, and no caller has to read a message to tell one from the other.
    values = grid("3.2:3.5:4")
    cell = next(one for one in Sweep(LOGISTIC, values, ("0.5",), 6).cells if one.value == values[1])
    refused = Failed(cell, "prompt is 32760 tokens; the context limit of 32768 leaves no room to generate")
    assert run_cell(Assorted({values[1]: outgrows(3)}), cell, 6) == refused
    assert run_cell(Assorted({values[1]: outgrows(0)}), cell, 6) == refused  # however early it came
    with pytest.raises(MapError, match="layer must be in"):
        run_cell(Assorted({values[1]: unmakeable}), cell, 6)
    with pytest.raises(SweepError, match="does not describe itself the same way twice"):
        run_cell(Assorted({values[1]: drifting}), cell, 6)
