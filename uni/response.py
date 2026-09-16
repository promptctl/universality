"""The model's own answer to a push along a direction: what the layers after the push write along it.

This is the loop's state as a number rather than a text. Push the residual stream `value` times a
direction, and read how far the stream sits along that direction some layers later, with no token
generated. Nothing is sampled, so the reading is a smooth function of the value, which a map read
off greedy text is not: there the text, and every number read off it, holds still until the value
crosses the point where one token overtakes another.
"""

from __future__ import annotations

from uni.model import Model, ModelError
from uni.steer import Steer


def response(model: Model, prompt: str, steer: Steer, value: float, layer: int) -> float:
    """How far the layers after the push, up to and including `layer`, move the stream along the direction."""
    turned = steer.turn(value)
    push = steer.direction.contrast.layer
    # [LAW:no-silent-failure] before the push the stream has not been pushed, and subtracting the
    # push from it would report a number the model never wrote.
    if layer < push:
        raise ModelError(f"the response is read at or after the layer the direction pushes, {push}; got {layer}")
    residual = model.prompt_residual(prompt, turned.additions, layer)
    # The residual connections carry the push itself to every later layer, so the stream's own
    # projection is the knob read back - `value` times the direction's squared length, a straight
    # line whatever the model does. What the model wrote is the stream less the push.
    pushed = sum(model.residual_vector(addition) for addition in turned.additions)
    return steer.direction.project(residual - pushed)
