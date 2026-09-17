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

import hashlib
import math
from collections.abc import Callable, Sequence
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


def read(model: Model, prompt: str, steer: Steer, value: float, layer: int, candidates: Callable[[torch.Tensor], torch.Tensor]) -> tuple[float, torch.Tensor, torch.Tensor]:
    """The answer to a push of `value` at `layer` with no token, the next token's log-probabilities, and the answer with each candidate appended.

    With a token the answer is read as `response` reads it, over one position more: the stream less
    the push at the prompt's tokens and at the token, averaged. So it is the response times the
    prompt's length, plus the token's own reading, over the length and one. [LAW:one-source-of-truth]
    the one reading both a draw's moments and a drawn token's answer are taken from.
    """
    admit(model, steer, value, layer)
    turned = steer.turn(value)
    pushed = sum(model.residual_vector(addition) for addition in turned.additions)
    stream, logprobs, own = model.next_tokens(prompt, turned.additions, layer, candidates, lambda residuals: steer.direction.along(residuals - pushed))
    length = stream.shape[0]
    response = steer.direction.project((stream - pushed).mean(dim=0))
    return response, logprobs, (length * response + own.cpu().double()) / (length + 1)


def drawn(model: Model, prompt: str, steer: Steer, value: float, layer: int, temperatures: Sequence[float]) -> Drawn:
    """The answer to a push of `value` at `layer`, without the next token and over its draw at each of `temperatures`.

    At a temperature T the token is drawn with probability proportional to p^(1/T), p its probability
    as the model gives it, and every token in the vocabulary is read.
    """
    response, logprobs, answers = read(model, prompt, steer, value, layer, lambda logprobs: torch.arange(len(logprobs)))
    draws = []
    for temperature in temperatures:
        weights = torch.softmax(logprobs / temperature, dim=0)
        mean = float(weights @ answers)
        draws.append(Draw(temperature, mean, math.sqrt(float(weights @ (answers - mean) ** 2))))
    return Drawn(response, tuple(draws))


def sampled(model: Model, prompt: str, steer: Steer, value: float, layer: int, temperature: float, key: bytes) -> float:
    """The answer to a push of `value` at `layer` with one token drawn after the prompt at `temperature`, the draw fixed by `key`.

    Fixed and not fresh, so a loop that draws is still a map from its state to the next one, and a
    trajectory written from it is the same bytes when run again: the key is what the caller holds
    the draw to, and a different key is an independent draw.
    """
    generator = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(key).digest()[:8], "little") >> 1)

    def draw(logprobs: torch.Tensor) -> torch.Tensor:
        return torch.multinomial(torch.softmax(logprobs / temperature, dim=0), 1, generator=generator)

    _, _, answers = read(model, prompt, steer, value, layer, draw)
    return float(answers[0])
