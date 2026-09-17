"""Whether readings taken many rows to a forward pass are the bits batch size one reads.

Every reading in this repo is taken a pass at a time, because a batch of prompts padded to one
length moves log-probabilities by about 1e-4 (the Determinism section). The batch the expensive
commands would use is not that one: one prompt repeated, every row the same length, differing only
in the push. This measures that batch against the readings taken alone - the answer `response`
reads and the next token's log-probabilities `next_tokens` reads - and builds nothing on the answer.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from uni.model import Model, token_logprobs
from uni.parse import ConfigError
from uni.response import admit, answered, response
from uni.steer import Steer

# How many times each batch size reads every push for its rate, the fastest kept: the machine is
# shared, and the slowest of three is the one another process ran through.
REPEATS = 3


class BatchError(ConfigError):
    """The batch sizes asked for cannot be measured over the pushes given. The message says why."""


@dataclass(frozen=True)
class Sizes:
    """The batch sizes to read the pushes at, each one a pass the pushes can fill, in increasing order.

    [LAW:parse-dont-validate] refused here, before a model is loaded: a size above the number of
    pushes runs one short pass and would be printed and timed as a batch it never was, and a largest
    size of one puts no row beside another, so reading it reversed is the same passes in another
    order and could only ever say "identical".
    """

    sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if list(self.sizes) != sorted(set(self.sizes)) or not self.sizes or self.sizes[0] < 1:
            raise BatchError(f"batch sizes are distinct, positive and increasing, got {self.sizes}")
        if self.largest < 2:
            raise BatchError("the largest batch size must be at least 2, so some pass holds a row beside another")

    @property
    def largest(self) -> int:
        return self.sizes[-1]

    @classmethod
    def over(cls, pushes: Sequence[float], sizes: Sequence[int]) -> Sizes:
        """The sizes asked for, refused unless every one of them is a pass the pushes fill."""
        over = sorted(size for size in set(sizes) if size > len(pushes))
        if over:
            raise BatchError(f"a batch of {', '.join(map(str, over))} rows is more than the {len(pushes)} pushes can fill; pass a size of at most {len(pushes)} or more pushes")
        return cls(tuple(sorted(set(sizes))))


@dataclass(frozen=True)
class Readings:
    """Every push's answer and next-token log-probabilities, in the order of the pushes."""

    answers: tuple[float, ...]
    logprobs: torch.Tensor  # shape (pushes, vocabulary), float64


@dataclass(frozen=True)
class Apart:
    """The largest distance between two readings of the same pushes, in the answer and in any token's log-probability."""

    answer: float
    logprob: float


def apart(one: Readings, other: Readings) -> Apart:
    """How far two readings of the same pushes are apart. Zero is the same bits: an equal float is never apart."""
    answer = max(abs(a - b) for a, b in zip(one.answers, other.answers, strict=True))
    return Apart(answer, float((one.logprobs - other.logprobs).abs().max()))


def alone(model: Model, prompt: str, steer: Steer, pushes: Sequence[float], layer: int) -> Readings:
    """Each push read as the repo reads it, a forward pass of its own."""
    # One candidate, the likeliest, because `next_tokens` reads at least one; its reading is not compared.
    def likeliest(logprobs: torch.Tensor) -> torch.Tensor:
        return logprobs.argmax().reshape(1)

    answers = tuple(response(model, prompt, steer, push, layer) for push in pushes)
    logprobs = [model.next_tokens(prompt, steer.turn(push).additions, layer, likeliest, lambda residuals: residuals[:, 0])[1] for push in pushes]
    return Readings(answers, torch.stack(logprobs))


def batched(model: Model, prompt: str, steer: Steer, pushes: Sequence[float], layer: int, size: int) -> Readings:
    """Every push read `size` rows to a forward pass, in order; the last pass takes what is left."""
    answers: list[float] = []
    logprobs = []
    for start in range(0, len(pushes), size):
        chunk = pushes[start : start + size]
        streams, logits = model.prompt_residuals(prompt, [steer.turn(push).additions for push in chunk], layer)
        # A row at a time, with the arithmetic a reading alone is taken with, so what differs is what the pass wrote.
        answers.extend(answered(model, steer, push, stream) for push, stream in zip(chunk, streams))
        logprobs.extend(token_logprobs(row) for row in logits)
    return Readings(tuple(answers), torch.stack(logprobs))


def rate(model: Model, prompt: str, steer: Steer, pushes: Sequence[float], layer: int, size: int) -> float:
    """Pushes read a second, `size` rows to a pass: the fastest of `REPEATS` readings of every push, after one to warm the kernels."""
    batched(model, prompt, steer, pushes[:size], layer, size)
    fastest = float("inf")
    for _ in range(REPEATS):
        torch.mps.synchronize()
        began = time.perf_counter()
        batched(model, prompt, steer, pushes, layer, size)
        torch.mps.synchronize()
        fastest = min(fastest, time.perf_counter() - began)
    return len(pushes) / fastest


@dataclass(frozen=True)
class Size:
    """One batch size, measured: how far its readings are from reading alone and from the largest batch, and how fast it reads."""

    size: int
    from_alone: Apart
    from_largest: Apart
    per_second: float


@dataclass(frozen=True)
class Measured:
    sizes: tuple[Size, ...]
    neighbours: Apart  # the largest batch read in reverse order against itself in order: whether a row's reading depends on the rows beside it


def measure(model: Model, prompt: str, steer: Steer, pushes: Sequence[float], layer: int, sizes: Sizes) -> Measured:
    """Every batch size's readings of the pushes against reading each alone, against the largest batch, and against the largest batch reversed."""
    for push in pushes:
        admit(model, steer, push, layer)
    reference = alone(model, prompt, steer, pushes, layer)
    readings = {size: batched(model, prompt, steer, pushes, layer, size) for size in sizes.sizes}
    largest = readings[sizes.largest]
    reversed_ = batched(model, prompt, steer, pushes[::-1], layer, sizes.largest)
    unreversed = Readings(reversed_.answers[::-1], reversed_.logprobs.flip(0))
    return Measured(
        tuple(Size(size, apart(readings[size], reference), apart(readings[size], largest), rate(model, prompt, steer, pushes, layer, size)) for size in sizes.sizes),
        apart(unreversed, largest),
    )
