"""The one writer, and the two things it promises: whole or absent, and never two runs' bytes."""

import os
from pathlib import Path

import pytest

from uni.atomic import write_whole


def scratch_names(monkeypatch, path, pids):
    """The scratch file each of those processes would write `path` through, rename withheld."""
    seen = []
    monkeypatch.setattr(Path, "replace", lambda self, target: seen.append(self))
    for pid in pids:
        monkeypatch.setattr("uni.atomic.os.getpid", lambda pid=pid: pid)
        write_whole(path, b"{}\n")
    return seen


def test_two_runs_writing_one_file_do_not_share_a_scratch_file(tmp_path, monkeypatch):
    # Rerunning a command is how work resumes here, so two runs reach the same file together.
    # Under one scratch name the second truncates the bytes the first is about to rename into
    # place, and a half-written file is renamed over the finished one.
    assert len(set(scratch_names(monkeypatch, tmp_path / "orbit.json", (111111, 999999)))) == 2


def test_a_write_killed_before_the_rename_leaves_the_earlier_file_whole(tmp_path, monkeypatch):
    # The rename is the whole promise: it is the one step that either happened or did not, so
    # everything before it can be interrupted without a reader ever seeing half a file.
    def killed(self, target):
        raise KeyboardInterrupt

    path = tmp_path / "orbit.json"
    path.write_bytes(b"the earlier file\n")
    monkeypatch.setattr(Path, "replace", killed)
    with pytest.raises(KeyboardInterrupt):
        write_whole(path, b"half of the next one\n")
    assert path.read_bytes() == b"the earlier file\n"
    assert list(tmp_path.glob("*.partial")) == []  # and takes its scratch file with it


def test_the_directory_a_file_goes_in_is_made(tmp_path):
    # A sweep's home and the trajectories directory are both made this way, so neither caller
    # carries its own mkdir for the writer to have to agree with.
    assert write_whole(tmp_path / "sweeps" / "abcd" / "orbit.json", b"{}\n").read_bytes() == b"{}\n"


def test_two_files_that_differ_only_by_extension_do_not_share_a_scratch_file(tmp_path, monkeypatch):
    # The scratch name is one file's, not one stem's. Put in place of the extension instead of
    # after the whole name, it would make `a.json` and `a.txt` write through each other - the
    # collision the process id is here to prevent, with the process id left out of it.
    seen = []
    monkeypatch.setattr(Path, "replace", lambda self, target: seen.append(self))
    for name in ("orbit.json", "orbit.txt"):
        write_whole(tmp_path / name, b"{}\n")
    assert len(set(seen)) == 2


def test_the_bytes_are_on_the_disk_before_the_rename(tmp_path, monkeypatch):
    # A rename can be durable while the bytes under it are not, and a file that is there is work
    # this program never does again - so a cell could end up finished and empty at once. The order
    # of the two is the whole of the promise, which is why it is pinned rather than assumed.
    order = []
    monkeypatch.setattr(os, "fsync", lambda fd: order.append("fsync"))
    monkeypatch.setattr(Path, "replace", lambda self, target: order.append("replace"))
    write_whole(tmp_path / "orbit.json", b"{}\n")
    assert order == ["fsync", "replace"]
