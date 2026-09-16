"""The steering knob, and the formality direction committed beside its contrast."""

import dataclasses
import hashlib
from itertools import islice

import pytest
import torch

from uni import steer
from uni.cli import build_parser
from uni.loop import orbit
from uni.maps import ModelMap, NoKnob
from uni.pinned import load_pinned
from uni.steer import Steer, SteerError, derive, load_contrast, read_direction
from uni.template import load_templates

TEXT = "The store will open late tomorrow because of the storm, so plan your trip around noon."


@pytest.fixture(scope="module")
def formality():
    return read_direction("formality", load_pinned())


def test_the_committed_direction_was_derived_from_the_committed_contrast(formality):
    assert formality.contrast == load_contrast("formality")


def test_a_direction_from_another_model_is_refused():
    other = dataclasses.replace(load_pinned(), revision="0" * 40)
    with pytest.raises(SteerError, match="run `uni direction formality`"):
        read_direction("formality", other)


def test_a_reply_that_encodes_to_nothing_is_refused(model):
    with pytest.raises(ValueError, match="encodes to no tokens"):
        model.reply_residual("hi", "", 12)


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
    assert Steer(formality).spec == {"kind": "steer", "direction": "formality", "layer": 12, "sha256": hashlib.sha256(file).hexdigest()}


def test_the_direction_is_what_its_pairs_produce(model, formality):
    # Close, not equal: machines differ in the sixth decimal of float32 kernels.
    derived = torch.tensor(derive(model, formality.contrast).vector)
    assert torch.allclose(derived, torch.tensor(formality.vector), atol=1e-4)


def rewrites(model, knob, value, steps=1):
    map = ModelMap(model, load_templates()["rewrite"], knob)
    return tuple(islice(orbit(map, value, TEXT), steps))


def test_zero_reproduces_the_unsteered_orbit(model, formality):
    assert rewrites(model, Steer(formality), 0.0, steps=3) == rewrites(model, NoKnob(), 0.0, steps=3)


def test_turning_the_value_moves_the_rewrite_along_the_direction(model, formality):
    contrast = formality.contrast
    direction = torch.tensor(formality.vector, device=model.device)

    def along(text):
        return float(model.reply_residual(contrast.template.render(TEXT), text, contrast.layer) @ direction)

    texts = [rewrites(model, Steer(formality), value)[0] for value in (-2.0, 0.0, 2.0)]
    assert len(set(texts)) == 3
    scores = [along(text) for text in texts]
    assert scores == sorted(scores), texts


def test_a_huge_value_runs(model, formality):
    assert len(rewrites(model, Steer(formality), 1e6)) == 1


def test_an_unknown_direction_is_refused_with_the_known_ones(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["loop", "--template", "rewrite", "--start", "x", "--steps", "1", "--knob", "nope"])
    assert "no nope.json in uni/directions; there are formality" in capsys.readouterr().err


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
