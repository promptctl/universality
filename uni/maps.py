"""The maps the loop runner iterates, and the knobs that turn the model's."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Protocol

from uni.parse import ConfigError
from uni.template import Template

if TYPE_CHECKING:  # the model is handed in; building one is the caller's cost
    from uni.model import Model, ResidualAdd


class MapError(ConfigError):
    """A map cannot be built as described, or cannot step the state it was handed."""


class KnobError(ConfigError):
    """A knob was asked for a value it cannot be turned to."""


@dataclass(frozen=True)
class Turned:
    """A knob at one setting: what it adds to the model, and what it was, for the trajectory file."""

    spec: Mapping[str, Any] | None  # None for the knob that does nothing
    additions: Sequence[ResidualAdd]


class Knob(Protocol):
    """A scalar the model can be turned by. Turning is by value, so one knob serves a whole sweep."""

    def turn(self, value: float) -> Turned: ...


class NoKnob:
    """The model unturned. A value has nothing to turn here, so only zero is a truthful one."""

    def turn(self, value: float) -> Turned:
        # [LAW:no-silent-failure] a trajectory recording a value nothing applied reads back as a steering run.
        if value:
            raise KnobError(f"there is no knob to turn, so the value must be 0, got {value}")
        return Turned(None, ())


@dataclass(frozen=True)
class ModelMap:
    """The pinned model under a template and a turned knob: the next state is the model's reply to the rendered state."""

    model: Model
    template: Template
    knob: Turned

    @property
    def spec(self) -> dict[str, Any]:
        return {
            "kind": "model",
            "template": {"name": self.template.name, "text": self.template.text},
            "pinned": asdict(self.model.pinned),
            "knob": self.knob.spec,
        }

    def step(self, state: str) -> str:
        # [LAW:dataflow-not-control-flow] every step adds the same additions; the unturned knob's are empty.
        return self.model.generate(self.template.render(state), self.knob.additions).text


def logistic_state(state: str) -> float:
    """The number a logistic state names, refused unless it is one the map holds.

    Written by the map and read back by the observable, so it is one function: a state the map
    could never have produced is one nothing downstream should be reading a number out of.
    """
    try:
        x = float(state)
    except ValueError as error:
        raise MapError(f"a logistic state is a number in 0..1, got {state!r}") from error
    # nan and the infinities fail this comparison rather than passing it, which is what is wanted.
    if not 0 <= x <= 1:
        raise MapError(f"a logistic state is a number in 0..1, got {state!r}")
    return x


@dataclass(frozen=True)
class Logistic:
    """x -> r x (1 - x) on [0, 1]: the map whose cascade every other map here is measured against.

    Textbook, and that is the point. Feigenbaum's period doubling is known for this map to more
    decimal places than this pipeline will ever resolve, so an orbit of it is the one fixture
    where a wrong answer is visible as a wrong answer rather than as a result.
    """

    r: float

    def __post_init__(self) -> None:
        # [LAW:parse-dont-validate] past this line a Logistic exists, and one exists only at an r
        # that keeps [0, 1] inside itself. Outside, the orbit runs to -inf and then to nan, and a
        # run of nans is a state that repeats: the detector would report period 1 on a divergence.
        if not 0 <= self.r <= 4:
            raise MapError(f"r must be in 0..4, or the map carries [0, 1] out of itself and the orbit diverges; got {self.r}")

    @property
    def spec(self) -> dict[str, Any]:
        # r fixes this orbit and is recorded as the trajectory's value, beside the spec and not in
        # it: one place, the same place a steering value is written. [LAW:one-source-of-truth]
        return {"kind": "logistic"}

    def step(self, state: str) -> str:
        x = logistic_state(state)
        # repr, which is the shortest text that reads back as exactly this float. The detector
        # compares the states themselves, so two spellings of one number would be two states.
        return repr(self.r * x * (1 - x))
