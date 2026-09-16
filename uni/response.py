"""The model's own answer to a push along a direction: what the layers after the push write along it.

This is the loop's state as a number rather than a text. Push the residual stream `value` times a
direction, and read how far the stream sits along that direction some layers later, with no token
generated. Nothing is sampled, so the reading is a smooth function of the value, which a map read
off greedy text is not: there the text, and every number read off it, holds still until the value
crosses the point where one token overtakes another.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from uni.model import Model, ModelError
from uni.steer import Steer

# How far rounding may move a reading before the reading is refused: a hundredth, the precision the
# results read off these curves are stated at - the hump's height, its curvature, a fit's residual.
TOLERANCE = 1e-2


def admit(model: Model, steer: Steer, value: float, layer: int) -> float:
    """The most rounding can move the answer to a push of `value` read at `layer`, refused unless it is a reading.

    Asked with no forward pass run, so a run can put every cell of its grid to it before the first
    one costs anything. [LAW:single-enforcer] `response` asks it too, so the two cannot disagree.

    The bound is the push's own. Every addition the stream makes while it holds the push rounds at
    the push's size, losing at most half the dtype's epsilon of it in each element: the push's own
    addition, the attention's and the MLP's in each layer after it, and the subtraction that takes
    it back out. Along the direction that is at most (additions / 2) * eps * sum |v_i| |push_i|.
    Past the tolerance the push has drowned what the model writes, and in the far limit rounded it
    away entirely, which reads as a model that answers nothing - a number, not an error.
    """
    push = steer.direction.contrast.layer
    last = len(model.layers) - 1
    # [LAW:no-silent-failure] before the push the stream has not been pushed, and subtracting the
    # push from it would report a number the model never wrote.
    if not push <= layer <= last:
        raise ModelError(f"the response is read at a layer from the one the direction pushes, {push}, to the model's last, {last}; got {layer}")
    direction = torch.tensor(steer.direction.vector).abs()
    eps = torch.finfo(model.dtype).eps
    bound = sum((layer - addition.layer + 1) * eps * float(direction @ addition.vector.abs().to(direction)) for addition in steer.turn(value).additions)
    # Written so that a nan bound, from a push no float holds, is refused with the rest.
    if not bound <= TOLERANCE:
        raise ModelError(
            f"{model.pinned.dtype} rounding could move the answer to a push of {value:g} at layer {layer} by {bound:.3g}, "
            f"past the {TOLERANCE} a reading is trusted to; a push this large drowns out what the model writes"
        )
    return bound


def response(model: Model, prompt: str, steer: Steer, value: float, layer: int) -> float:
    """How far the layers after the push, up to and including `layer`, move the stream along the direction."""
    admit(model, steer, value, layer)
    turned = steer.turn(value)
    stream = model.prompt_residual(prompt, turned.additions, layer)
    # The residual connections carry the push itself to every later layer, so the stream's own
    # projection is the knob read back - `value` times the direction's squared length, a straight
    # line whatever the model does. What the model wrote is the stream less the push, taken out
    # token by token and only then averaged, as `admit` counts it.
    pushed = sum(model.residual_vector(addition) for addition in turned.additions)
    return steer.direction.project((stream - pushed).mean(dim=0))


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
