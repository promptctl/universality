"""What temperature puts into the response map: the answer read with the token the model writes after the prompt.

The response map generates nothing, so its answer is a smooth function of the push. A loop that
generates samples, and the least it can generate is one token: the model's next token under the
push, drawn at a temperature, with the answer read over the prompt and that token together. The
answer then depends on which token was drawn. Over the draw it has a mean, which is a map, and a
spread, which is noise that map is read with at every step: PROJECT.md's "temperature is noise",
as a number.

Both are taken over every token in the vocabulary rather than over a sample of them, so neither
carries a sampling error. It matters where the loop's top is: pushed there, the next token is so
evenly spread that at temperature 1 the 16384 likeliest tokens hold 97% of the probability.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from uni.model import Model
from uni.response import admit
from uni.steer import Steer


@dataclass(frozen=True)
class Draw:
    """The answer to a push read with the token after the prompt, over that token's draw at one temperature."""

    temperature: float
    mean: float
    spread: float  # the standard deviation over the draw


@dataclass(frozen=True)
class Drawn:
    """The answer to a push with no token written, as `response` reads it, and with one drawn at each temperature."""

    response: float
    draws: tuple[Draw, ...]


def drawn(model: Model, prompt: str, steer: Steer, value: float, layer: int, temperatures: Sequence[float]) -> Drawn:
    """The answer to a push of `value` at `layer`, without the next token and over its draw at each of `temperatures`.

    With a token the answer is read as `response` reads it, over one position more: the stream less
    the push at the prompt's tokens and at the token, averaged. So it is the response times the
    prompt's length, plus the token's own reading, over the length and one. At a temperature T the
    token is drawn with probability proportional to p^(1/T), p its probability as the model gives it.
    """
    admit(model, steer, value, layer)
    turned = steer.turn(value)
    pushed = sum(model.residual_vector(addition) for addition in turned.additions)
    stream, logprobs, own = model.next_tokens(prompt, turned.additions, layer, lambda residuals: steer.direction.along(residuals - pushed))
    length = stream.shape[0]
    response = steer.direction.project((stream - pushed).mean(dim=0))
    answers = (length * response + own.cpu().double()) / (length + 1)
    draws = []
    for temperature in temperatures:
        weights = torch.softmax(logprobs / temperature, dim=0)
        mean = float(weights @ answers)
        draws.append(Draw(temperature, mean, math.sqrt(float(weights @ (answers - mean) ** 2))))
    return Drawn(response, tuple(draws))
