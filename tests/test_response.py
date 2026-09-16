"""The model's own response to a push along the formality direction, on the pinned model."""

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


def test_a_layer_before_the_push_is_refused(model, formality):
    with pytest.raises(ModelError, match="at or after the layer the direction pushes, 12"):
        response(model, PROMPT, formality, 1.0, 11)
