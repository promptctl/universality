"""The model's own answer to a push along a direction: what the layers after the push write along it.

This is the loop's state as a number rather than a text. Push the residual stream `value` times a
direction, and read how far the stream sits along that direction some layers later, with no token
generated. Nothing is sampled, so the reading is a smooth function of the value, which a map read
off greedy text is not: there the text, and every number read off it, holds still until the value
crosses the point where one token overtakes another.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

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
    answer = steer.direction.project(residual - pushed)
    # [LAW:no-silent-failure] as in generate and reply_logprob: a push large enough to overflow the
    # forward pass gives a nan, and every comparison with a nan is false, so it would be printed as
    # one more reading and pass through the maximum and the turns unremarked.
    if not math.isfinite(answer):
        raise ModelError(f"the answer to a push of {value} has no finite projection; the push overflows this model's arithmetic")
    return answer


def turns(values: Sequence[float], readings: Sequence[float]) -> tuple[float, ...]:
    """The values at which the curve stops rising and falls, or stops falling and rises.

    A hump is one of these with the curve falling on either side. Counted on the grid as given,
    with nothing smoothed away, so a curve that jitters says so. A flat step is not a turn and does
    not hide one: two equal readings at a top still leave the curve rising before them and falling
    after, so each slope is compared with the last one that was not flat, and the turn is placed
    where that slope ended.
    """
    found = []
    last: tuple[bool, int] | None = None  # whether the last slope that moved rose, and the reading it ended at
    for end, (before, after) in enumerate(zip(readings, readings[1:]), start=1):
        if after == before:
            continue
        rising = after > before
        if last is not None and rising != last[0]:
            found.append(values[last[1]])
        last = (rising, end)
    return tuple(found)
