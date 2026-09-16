"""The CLI's error contract: the three ways a run can stop, and how each one reads.

`ConfigError` is the one this program raises about itself, and the rest of the suite is full of
it. The other two belong to the world - a machine that will not do the work, and a consumer that
stopped reading - and neither is a bug here, so neither should arrive as a traceback.
"""

import errno
import os
import subprocess
import sys
from pathlib import Path

from uni.cli import EXIT_BUG, EXIT_CONFIG, EXIT_DIVERGED, EXIT_INCOMPLETE, EXIT_IO, EXIT_PIPE, main

SWEEP = ["sweep", "--map", "logistic", "--grid", "3.2:3.5:4", "--start", "0.5", "--steps", "3"]


def command(argv):
    return main(argv, {}, Path.cwd())


def test_a_machine_that_will_not_do_the_work_is_reported_and_not_raised(capsys, tmp_path, monkeypatch):
    # A traceback means a bug in this program, and a sweeps directory somebody mangled by hand is
    # not one. The path here is one `uni` computed rather than one the user typed, so nothing
    # upstream catches it: before this it reached the user as a traceback, and exited 1 - which is
    # EXIT_DIVERGED - so a crash and a determinism failure read alike to anything reading the code.
    monkeypatch.setattr("uni.cli.SWEEPS", tmp_path)
    assert command([*SWEEP, "--status"]) == 0
    home = Path(capsys.readouterr().out.splitlines()[0].split("sweep ", 1)[1])
    home.write_text("a file standing where this sweep keeps its work")
    assert command(SWEEP) == EXIT_IO
    printed = capsys.readouterr()
    assert printed.err.startswith("uni: ")  # the shape every other refusal here has
    assert str(home) in printed.err  # which file is the whole of what a person fixes
    assert "File exists" in printed.err  # in the OS's own words, because they are the accurate ones


def into_a_closed_pipe(cwd, *args, stderr):
    """Run `uni` as its own process, its stdout a pipe whose reader has gone - `| head` once head
    has exited. A process, because what interpreter shutdown does to the exit code is only visible
    from outside one."""
    read, write = os.pipe()
    os.close(read)
    command = [sys.executable, "-c", "from uni.cli import entry; entry()", *args]
    ran = subprocess.run(command, cwd=cwd, stdout=write, stderr=stderr, text=True)
    os.close(write)
    return ran


def test_a_consumer_that_stops_reading_is_not_a_failure(tmp_path):
    # `uni sweep ... | head` is ordinary shell usage, and every command here prints, so every one
    # of them met this as a traceback followed by Python's own "Exception ignored in" on the way
    # out. Nothing about the run failed, so there is nothing to report - and the code says what a
    # shell says of a producer killed by the same event, rather than claiming a fault.
    ran = into_a_closed_pipe(tmp_path, *SWEEP, stderr=subprocess.PIPE)
    assert (ran.returncode, ran.stderr) == (EXIT_PIPE, "")


def test_a_refusal_nobody_is_left_to_read_still_exits_as_a_refusal(tmp_path):
    # `2>&1 | head`: the explanation has nowhere to go, and the answer still does. This exited 120
    # when saying why raised in place of returning.
    ran = into_a_closed_pipe(tmp_path, "loop", "--map", "logistic", "--start", "0.5", "--steps", "5", stderr=subprocess.STDOUT)
    assert ran.returncode == EXIT_CONFIG


def test_help_nobody_is_left_to_read_still_exits_as_help(tmp_path):
    # argparse writes --help and its usage errors and leaves through SystemExit, past everything
    # `main` does; this exited 120 with "Exception ignored in" on the way out.
    assert into_a_closed_pipe(tmp_path, "--help", stderr=subprocess.STDOUT).returncode == 0
    assert into_a_closed_pipe(tmp_path, "sweep", "--grd", "1", stderr=subprocess.STDOUT).returncode == 2


def test_a_bug_here_exits_as_a_bug_and_not_as_an_answer(tmp_path):
    # The interpreter exits 1 for an exception nothing caught, and 1 is EXIT_DIVERGED: a crash
    # partway through a run read as the determinism gate's verdict to anything switching on it.
    broken = "import uni.cli as cli; cli.run = lambda *_: 1 / 0; cli.entry()"
    ran = subprocess.run([sys.executable, "-c", broken], cwd=tmp_path, capture_output=True, text=True)
    assert ran.returncode == EXIT_BUG
    assert "ZeroDivisionError" in ran.stderr  # still a traceback: the code changed, not the report


class Full:
    """A stdout on a disk with no room left: it takes the text, and fails to deliver it."""

    def write(self, text):
        return len(text)

    def flush(self):
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))


def test_output_the_disk_could_not_take_is_not_a_success(capsys, monkeypatch):
    # `uni host` prints without flushing, so its only write is the last one. Left to interpreter
    # exit, a disk refusing it read as exit 120 and a message about a TextIOWrapper.
    monkeypatch.setattr(sys, "stdout", Full())
    assert command(["host"]) == EXIT_IO
    assert os.strerror(errno.ENOSPC) in capsys.readouterr().err


def test_output_that_could_not_be_delivered_does_not_outrank_an_answer(monkeypatch):
    # A sweep come up short whose count line never arrived is still a sweep come up short: a resume
    # loop switching on 79 must see 79, not the disk's 74 or the pipe's 141.
    monkeypatch.setattr("uni.cli.run", lambda *_: EXIT_INCOMPLETE)
    monkeypatch.setattr(sys, "stdout", Full())
    assert command(["host"]) == EXIT_INCOMPLETE


def test_a_stdout_closed_before_the_run_is_output_to_nowhere(tmp_path):
    # `uni host >&-`: Python hands such a process a stdout of None, which print tolerates and
    # nothing else did - this left with a traceback and exit 1, which is EXIT_DIVERGED.
    command = [sys.executable, "-c", "from uni.cli import entry; entry()", "host"]
    ran = subprocess.run(command, cwd=tmp_path, preexec_fn=lambda: os.close(1), stderr=subprocess.PIPE, text=True)
    assert (ran.returncode, ran.stderr) == (0, "")


def test_each_way_a_run_can_end_has_its_own_code():
    # [LAW:one-source-of-truth] one number per answer, or a reader switching on the code - which is
    # what an unattended run does - cannot tell two of them apart. This is the check that would
    # have caught what this ticket is about: an uncaught OSError left the interpreter with 1.
    codes = (0, EXIT_DIVERGED, EXIT_CONFIG, EXIT_INCOMPLETE, EXIT_IO, EXIT_PIPE, EXIT_BUG, 2)  # 2 is argparse's
    assert len(set(codes)) == len(codes)
