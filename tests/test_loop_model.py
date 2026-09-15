"""The loop on the pinned model, through the library and through `uni loop` in fresh processes."""

import subprocess
import sys
from itertools import islice

import pytest

from uni.loop import Trajectory, orbit, read_trajectory
from uni.maps import ModelMap
from uni.template import load_templates

PARAGRAPH = "The lighthouse keeper climbed the stairs each night, counting them aloud so the dark would not feel so large."


@pytest.fixture(scope="module")
def templates():
    return load_templates()


def trajectory(model, template, start, steps):
    map = ModelMap(model, template)
    return Trajectory(map.spec, 0.0, start, tuple(islice(orbit(map, 0.0, start), steps)))


def test_identity_template_repeats_the_start(model, templates):
    assert trajectory(model, templates["identity"], "hello", 10).states == ("hello",) * 10


def test_rewrite_twice_encodes_the_same_bytes(model, templates):
    first, second = (trajectory(model, templates["rewrite"], PARAGRAPH, 3) for _ in range(2))
    assert first.states[0] != PARAGRAPH  # the rewrite did something
    assert first.encode() == second.encode()


def test_empty_template_and_empty_start_make_a_trajectory(model, templates):
    assert len(trajectory(model, templates["empty"], "", 2).states) == 2


def test_the_trajectory_records_the_template_and_pinned_config(model, templates):
    spec = trajectory(model, templates["rewrite"], "hi", 1).map
    assert spec["template"]["text"] == templates["rewrite"].text
    assert spec["pinned"]["revision"] == model.pinned.revision


def uni_loop(cwd, *args):
    command = [sys.executable, "-c", "from uni.cli import entry; entry()", "loop", *args]
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=True).stdout


def test_the_command_prints_every_state_and_rewrites_the_same_file(tmp_path):
    args = ("--template", "identity", "--steps", "10", "--start", "hello")
    first = uni_loop(tmp_path, *args)
    path = tmp_path / first.split()[-1]
    written = path.read_bytes()
    assert uni_loop(tmp_path, *args) == first
    assert path.read_bytes() == written
    assert first.splitlines()[2:12] == [f"{step:>4}  'hello'" for step in range(1, 11)]
    assert read_trajectory(path).states == ("hello",) * 10
