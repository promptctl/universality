"""The loop on the pinned model, through the library and through `uni loop` in fresh processes."""

import subprocess
import sys
from itertools import islice

import pytest

from uni.loop import Trajectory, orbit, read_trajectory
from uni.maps import MapError, ModelFamily, ModelMap, NoKnob, Turned, model_spec
from uni.model import ModelError, ResidualAdd
from uni.template import load_templates

PARAGRAPH = "The lighthouse keeper climbed the stairs each night, counting them aloud so the dark would not feel so large."


@pytest.fixture(scope="module")
def templates():
    return load_templates()


def trajectory(model, template, start, steps):
    map = ModelMap(model, template, NoKnob().turn(0.0))
    return Trajectory(map.spec, 0.0, start, tuple(islice(orbit(map, start), steps)))


def test_identity_template_repeats_the_start(model, templates):
    assert trajectory(model, templates["identity"], "hello", 10).states == ("hello",) * 10


def test_rewrite_twice_encodes_the_same_bytes(model, templates):
    first, second = (trajectory(model, templates["rewrite"], PARAGRAPH, 3) for _ in range(2))
    assert first.states[0] != PARAGRAPH  # the rewrite did something
    assert first.encode() == second.encode()


def test_empty_template_and_empty_start_make_a_trajectory(model, templates):
    assert len(trajectory(model, templates["empty"], "", 2).states) == 2


def test_a_reply_the_budget_cut_off_is_not_a_state(model, templates):
    # universality-sweep-zjh: a steered sweep's wings filled the token budget every step, and those
    # cuts were recorded as the model's states. Four tokens of room makes the cut certain.
    from uni.determinism import context_limit_prompt

    map = ModelMap(model, templates["empty"], NoKnob().turn(0.0))
    with pytest.raises(MapError, match="did not end within 4 tokens"):
        map.step(context_limit_prompt(model, 4))


def test_an_orbit_recorded_while_cuts_were_kept_is_not_named_as_one_of_this_map(model, templates):
    # Sweeps resume by name, so an old orbit holding the budget's cuts as states would otherwise
    # count as a finished cell of a map that refuses them.
    spec = model_spec(model.pinned, templates["rewrite"], None)
    kept = {key: value for key, value in spec.items() if key != "truncated"}
    assert Trajectory(kept, 0.0, "hi", ("a",)).name != Trajectory(spec, 0.0, "hi", ("a",)).name


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


def test_a_family_reads_the_checkpoint_once_and_not_until_a_cell_runs(monkeypatch, templates):
    # The claim a sweep of a thousand cells rests on: one checkpoint per family, not one per cell
    # - and none at all for a family that is only ever asked what it is, which is what `uni sweep
    # --status` asks. Stubbed rather than loaded, because what is under test is how many times the
    # loading happens and not what it returns.
    from uni.maps import ModelFamily
    from uni.pinned import load_pinned

    loaded = []
    monkeypatch.setattr("uni.model.Model", lambda pinned: loaded.append(pinned) or "the checkpoint")
    family = ModelFamily(load_pinned(), templates["rewrite"], NoKnob())
    assert family.spec["pinned"]["revision"] == load_pinned().revision
    assert loaded == []  # naming the map cost nothing
    first, second = family.at(0.0), family.at(2.0 - 2.0)
    assert (first.model, second.model) == ("the checkpoint", "the checkpoint")
    assert len(loaded) == 1


def test_a_family_refuses_a_start_the_context_has_no_room_to_answer(model, templates, monkeypatch):
    # The states a model map cannot step are the ones that leave the context no room for a reply,
    # and that is the model's own arithmetic rather than a second copy of it in the family. Asked
    # of the checkpoint the run is about to load anyway, so a sweep does not write its manifest,
    # load the model, die on its first cell, and die there again on every resume.
    from uni.maps import ModelFamily
    from uni.model import ModelError

    monkeypatch.setattr("uni.model.Model", lambda pinned: model)
    family = ModelFamily(model.pinned, templates["rewrite"], NoKnob())
    family.holds(("a start the context has plenty of room to answer",))
    with pytest.raises(ModelError, match="leaves no room to generate"):
        family.holds(("word " * model.context_limit,))


class Elsewhere:
    """A knob that adds to a layer no checkpoint this size has, as a hand-edited direction does."""

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    @property
    def spec(self):
        return {"kind": "elsewhere"}

    def turn(self, value):
        import torch

        return Turned(value, self.spec, (ResidualAdd(999, torch.zeros(self.hidden_size)),))


def test_a_family_whose_knob_does_not_fit_the_checkpoint_makes_no_map(model, templates):
    # A direction's layer and the length of its vector are fixed by the direction, not by the
    # setting, so an addition this checkpoint cannot take is one no cell of a sweep could have
    # taken. Caught where the map is made, one sweep of six hundred cells is refused once, at the
    # first cell, before a token is generated; caught in the decoding loop, it was refused six
    # hundred times by a run that could never have written a file.
    family = ModelFamily(model.pinned, templates["identity"], Elsewhere(model.hidden_size))
    family.__dict__["model"] = model  # the checkpoint this session already holds, not a second one
    assert family.model is model  # said out loud, so a family that stops caching fails here
    # rather than quietly loading half a billion parameters a second time.
    with pytest.raises(ModelError, match="layer must be in"):
        family.at(0.0)


def test_a_family_whose_knob_fits_makes_a_map_whose_additions_are_on_the_device(model, templates):
    # The other half: what `at` hands back is a map whose additions have already landed where they
    # are added, so the vector reaches the device once per map rather than once per step.
    import torch

    class Fitting(Elsewhere):
        def turn(self, value):
            return Turned(value, self.spec, (ResidualAdd(0, value * torch.ones(self.hidden_size)),))

    family = ModelFamily(model.pinned, templates["identity"], Fitting(model.hidden_size))
    family.__dict__["model"] = model
    assert family.model is model
    (addition,) = family.at(2.0).knob.additions
    assert addition.vector.device.type == model.device.type
    assert addition.vector.dtype == model.dtype
