"""The model's own response to a push along the formality direction, on the pinned model."""

from pathlib import Path

import pytest

from uni.model import ModelError
from uni.response import response
from uni.steer import Steer, read_direction
from uni.template import load_templates

PROMPT = load_templates()["rewrite"].render("The meeting moved to Thursday because the room was booked.")


@pytest.fixture(scope="module")
def formality(model):
    return Steer(read_direction("formality", model.pinned))


def test_at_the_layer_it_is_pushed_the_stream_has_not_answered_the_push(model, formality):
    # Nothing after the push has run, so what is left once the push is taken away is the same
    # whatever the push was. It is also the test that the reading holds the push at all: a reader
    # that ran before the addition would come out 10 * |v|^2 lower at 10.
    layer = formality.direction.contrast.layer
    assert response(model, PROMPT, formality, 10.0, layer) == pytest.approx(response(model, PROMPT, formality, 0.0, layer), abs=1e-2)


def test_a_few_layers_on_the_response_rises_to_one_maximum_and_falls(model, formality):
    # Measured over x in [-40, 40] at 321 points (universality-rung1-7er): at layer 16 the response
    # climbs from the negative end to a single maximum near x = -8.75 and falls past it. Pinned as
    # the ordering that makes it a hump, not as the digits.
    at = {x: response(model, PROMPT, formality, x, 16) for x in (-40.0, -10.0, 0.0, 10.0)}
    assert at[-40.0] < at[-10.0] > at[0.0] > at[10.0]


@pytest.mark.parametrize("layer", [11, 24])
def test_a_layer_before_the_push_or_past_the_model_is_refused(model, formality, layer):
    with pytest.raises(ModelError, match=f"from the one the direction pushes, 12, to the model's last, 23; got {layer}"):
        response(model, PROMPT, formality, 1.0, layer)


def test_turns_are_where_the_slope_changes_sign_on_the_grid_as_given():
    from uni.response import turns

    assert turns((0, 1, 2, 3, 4), (0, 2, 3, 1, 5)) == (2, 3)
    assert turns((0, 1, 2), (0, 1, 2)) == ()
    # A top two readings wide is still a top, placed where the rise ended.
    assert turns((0, 1, 2, 3), (0, 1, 1, 0)) == (1,)
    assert turns((0, 1, 2, 3, 4), (3, 1, 1, 1, 2)) == (1,)


def test_a_push_that_drowns_what_the_model_writes_is_refused_rather_than_read(model, formality):
    # Found in review: at 1e12 the model's writes are rounded out of the stream, and what is left once
    # the push is taken back out read -0.0003 - an answer of nothing, printed as one more reading.
    with pytest.raises(ModelError, match="rounding could move the answer to a push of 1e\\+12 at layer 23"):
        response(model, PROMPT, formality, 1e12, 23)
    with pytest.raises(ModelError, match="rounding"):
        response(model, PROMPT, formality, float("nan"), 23)


def test_the_measured_grid_is_read_well_inside_the_tolerance(model, formality):
    from uni.response import TOLERANCE, admit

    # The bound grows with the push and with the layers the push passes through, so the grid's far
    # end at the last layer is its worst cell.
    assert admit(model, formality, -40.0, 23) < TOLERANCE / 10
    assert admit(model, formality, 0.0, 23) == 0.0


def test_a_run_refuses_a_cell_before_it_reads_any(model, monkeypatch, capsys):
    # Found in review: a layer past the model was met only after every layer before it had been
    # read across the whole grid, and the run died having printed nothing.
    from uni.cli import EXIT_CONFIG, main

    monkeypatch.setattr("uni.model.Model", lambda pinned: model)
    monkeypatch.setattr(model, "prompt_residual", lambda *_: pytest.fail("read a cell of a run that was going to be refused"))
    argv = ["response", "--template", "rewrite", "--knob", "formality", "--start", "a", "--grid=-40:40:3", "--layer", "16", "--layer", "30"]
    assert main(argv, {}, Path.cwd()) == EXIT_CONFIG
    assert "got 30" in capsys.readouterr().err
