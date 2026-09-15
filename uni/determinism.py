"""The determinism gate: the same generation, run many times, must hash the same every time."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # the CLI reads RUNS for its help text without loading torch
    from uni.model import Model

RUNS = 100  # PROJECT.md, Rung 0
ROOM = 16  # tokens the context-limit case leaves free, so its generation ends at the limit


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str


def context_limit_prompt(model: Model, room: int) -> str:
    """A prompt that fills the context limit but for `room` tokens, template included."""
    overhead = model.encode("").shape[1]
    # " hello" is one token each, and "hello" is one token after the template's opening.
    prompt = "hello" + " hello" * (model.context_limit - room - overhead - 1)
    length = model.encode(prompt).shape[1]
    if length != model.context_limit - room:
        raise RuntimeError(f"context-limit prompt is {length} tokens, wanted {model.context_limit - room}; the tokenizer merged the filler")
    return prompt


def cases(model: Model) -> tuple[Case, ...]:
    return (
        Case("ordinary", "In two sentences, why is the sky blue?"),  # not the empty prompt's greeting
        Case("empty", ""),
        Case("context-limit", context_limit_prompt(model, ROOM)),
    )


def hashes(model: Model, case: Case, runs: int) -> Iterator[str]:
    # A generator, so a caller can show each run as it lands; a full-context run takes seconds.
    return (model.generate(case.prompt).sha256 for _ in range(runs))
