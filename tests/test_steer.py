"""The steering knob, and the formality direction committed beside its contrast."""

import dataclasses
import hashlib
from itertools import islice
from pathlib import Path

import pytest
import torch

from uni import steer
from uni.cli import EXIT_CONFIG, build_parser, main
from uni.loop import orbit
from uni.maps import KnobError, ModelMap, NoKnob
from uni.model import ModelError
from uni.pinned import load_pinned
from uni.steer import Steer, SteerError, derive, load_contrast, read_direction
from uni.template import load_templates

TEXT = "The store will open late tomorrow because of the storm, so plan your trip around noon."


@pytest.fixture(scope="module")
def formality():
    return read_direction("formality", load_pinned())


def test_the_committed_direction_was_derived_from_the_committed_contrast(formality):
    assert formality.contrast == load_contrast("formality")


def elsewhere(tmp_path, monkeypatch):
    """The committed direction and its contrast in a directory a test may edit."""
    for name in ("formality.json", "formality.toml"):
        (tmp_path / name).write_bytes((steer.DIRECTIONS / name).read_bytes())
    monkeypatch.setattr(steer, "DIRECTIONS", tmp_path)
    return tmp_path


def test_a_direction_file_that_was_edited_is_refused(tmp_path, monkeypatch):
    files = elsewhere(tmp_path, monkeypatch)
    (files / "formality.json").write_bytes((files / "formality.json").read_bytes().replace(b"\n  ", b"\n "))
    with pytest.raises(SteerError, match="re-derive it rather than editing it"):
        read_direction("formality", load_pinned())


def test_a_direction_that_no_longer_matches_its_contrast_is_refused(tmp_path, monkeypatch):
    files = elsewhere(tmp_path, monkeypatch)
    (files / "formality.toml").write_text((files / "formality.toml").read_text().replace("layer = 12", "layer = 14"))
    with pytest.raises(SteerError, match="no longer matches the contrast"):
        read_direction("formality", load_pinned())


@pytest.mark.parametrize("vector", ["[]", "[1.0, NaN]"])
def test_a_direction_whose_vector_is_unusable_is_refused(tmp_path, monkeypatch, vector):
    files = elsewhere(tmp_path, monkeypatch)
    text = (files / "formality.json").read_text()
    head, _, tail = text.partition('  "vector": [')
    (files / "formality.json").write_text(head + '  "vector": ' + vector + "\n}\n")
    with pytest.raises(SteerError, match="only finite floats"):
        read_direction("formality", load_pinned())


def test_a_longer_generation_limit_leaves_a_direction_fresh(formality):
    longer = dataclasses.replace(load_pinned(), max_new_tokens=load_pinned().max_new_tokens * 2)
    assert read_direction("formality", longer) == formality


def test_a_direction_from_another_model_is_refused():
    other = dataclasses.replace(load_pinned(), revision="0" * 40)
    with pytest.raises(SteerError, match="run `uni direction formality`"):
        read_direction("formality", other)


def test_a_reply_that_encodes_to_nothing_is_refused(model):
    with pytest.raises(ModelError, match="encodes to no tokens"):
        model.reply_residual("hi", "", 12)


def test_a_prompt_and_reply_that_cannot_be_held_together_are_refused(model):
    # A reply the model could write against a nearly full context cannot then be scored beside it.
    with pytest.raises(ModelError, match="context limit"):
        model.encode_reply("hi " * model.context_limit, "hello")


def test_a_reply_scored_under_a_collapsing_value_is_refused(model, formality):
    # The value is a field of the trajectory file, so the observables meet one the run never did.
    additions = Steer(formality).turn(1e300).additions
    with pytest.raises(ModelError, match="no finite log-probability"):
        model.reply_logprob(formality.contrast.template.render(TEXT), "Something.", additions)


def test_the_reply_span_is_the_reply_alone(model):
    ids, span = model.encode_reply("hi", "Hello there.")
    assert model.tokenizer.decode(ids[0, span]) == "Hello there."


# Written as powers, not as words: the run host's name is a word this repo must never carry.
@pytest.mark.parametrize("value", ["nan", "1e999", "-1e999"])
def test_a_value_that_is_not_finite_is_refused(capsys, value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["loop", "--template", "rewrite", "--start", "x", "--steps", "1", f"--value={value}"])
    assert "must be a finite number" in capsys.readouterr().err


def test_the_trajectory_names_the_exact_direction_file(formality):
    file = (steer.DIRECTIONS / "formality.json").read_bytes()
    spec = Steer(formality).turn(2.0).spec
    assert spec == {"kind": "steer", "direction": "formality", "layer": 12, "sha256": hashlib.sha256(file).hexdigest()}


def test_the_projection_is_taken_at_the_precision_the_vector_is_kept_at(formality):
    # A checkpoint pinned at float16 would otherwise round the vector before the dot product,
    # and the reading is printed to six decimals the rounding would own.
    coarse = dataclasses.replace(formality, vector=(1.0001,) * 8)
    assert coarse.project(torch.ones(8, dtype=torch.float16)) == pytest.approx(8.0008, rel=1e-6)


def test_the_direction_is_what_its_pairs_produce(model, formality):
    # Close, not equal: machines differ in the sixth decimal of float32 kernels.
    derived = torch.tensor(derive(model, formality.contrast).vector)
    assert torch.allclose(derived, torch.tensor(formality.vector), atol=1e-4)


def rewrites(model, knob, value, steps=1):
    map = ModelMap(model, load_templates()["rewrite"], knob.turn(value))
    return tuple(islice(orbit(map, TEXT), steps))


def test_zero_reproduces_the_unsteered_orbit(model, formality):
    assert rewrites(model, Steer(formality), 0.0, steps=3) == rewrites(model, NoKnob(), 0.0, steps=3)


def test_turning_the_value_moves_the_rewrite_along_the_direction(model, formality):
    contrast = formality.contrast
    direction = torch.tensor(formality.vector, device=model.device)

    def along(text):
        return float(model.reply_residual(contrast.template.render(TEXT), text, contrast.layer) @ direction)

    # Inside the range where a rewrite of TEXT ends by itself: at 2.0 it runs to the token budget,
    # which the map refuses rather than records. [universality-sweep-zjh]
    texts = [rewrites(model, Steer(formality), value)[0] for value in (-1.5, 0.0, 1.5)]
    assert len(set(texts)) == 3
    scores = [along(text) for text in texts]
    assert scores == sorted(scores), texts


# Measured, not assumed: up here rsqrt of an overflowed variance zeroes the layer, so the
# logits stay finite and the text stays real text. Only an infinite addition collapses them.
# Asked of the model and not of a map, because what is claimed is that the forward pass survives:
# the reply it writes up here never ends, and a map refuses that as a state.
@pytest.mark.parametrize("value", [1e6, 1e20, 1e38])
def test_a_huge_value_still_generates(model, formality, value):
    prompt = load_templates()["rewrite"].render(TEXT)
    assert model.generate(prompt, Steer(formality).turn(value).additions).text


def test_a_value_that_collapses_the_forward_pass_is_refused(model, formality):
    with pytest.raises(ModelError, match="no finite logits"):
        rewrites(model, Steer(formality), 1e300)


def test_a_value_with_no_knob_to_turn_is_refused():
    with pytest.raises(KnobError, match="no knob to turn"):
        NoKnob().turn(2.0)


def test_a_value_with_no_knob_is_refused_before_the_checkpoint_is_read(capsys, monkeypatch):
    monkeypatch.setattr("uni.pinned.load_pinned", lambda: pytest.fail("the checkpoint was read before the value was refused"))
    argv = ["loop", "--template", "rewrite", "--start", "x", "--steps", "1", "--value", "2"]
    assert main(argv, {}, Path.cwd()) == EXIT_CONFIG
    assert "uni: there is no knob to turn" in capsys.readouterr().err


def test_an_unknown_direction_is_refused_with_the_known_ones(capsys):
    # Read by the map that owns --knob rather than by argparse, so it is the CLI's own refusal:
    # `uni: ...` and EXIT_CONFIG, not argparse's exit 2.
    argv = ["loop", "--template", "rewrite", "--start", "x", "--steps", "1", "--knob", "nope"]
    assert main(argv, {}, Path.cwd()) == EXIT_CONFIG
    assert "no nope.json in uni/directions; there are brevity, certainty, formality, past, positivity" in capsys.readouterr().err


PAIR = '[[pairs]]\ntext = "a"\ntoward = "b"\naway = "c"\n'


@pytest.mark.parametrize(
    "text, message",
    [
        ('template = "nope"\nlayer = 1\n' + PAIR, "no template 'nope'"),
        ('template = "rewrite"\nlayer = -1\n' + PAIR, "layer must not be negative"),
        ('template = "rewrite"\nlayer = 1\npairs = []\n', "pairs is empty"),
        ('template = "rewrite"\nlayer = 1\n[[pairs]]\ntext = "a"\ntoward = "b"\n', "away is missing"),
        ('template = "rewrite"\nlayer = [', "does not parse"),
    ],
)
def test_a_malformed_contrast_is_refused(tmp_path, monkeypatch, text, message):
    monkeypatch.setattr(steer, "DIRECTIONS", tmp_path)
    (tmp_path / "bad.toml").write_text(text)
    with pytest.raises(SteerError, match=message):
        load_contrast("bad")
