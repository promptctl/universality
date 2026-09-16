"""The maps the loop runner iterates, and the knobs that turn the model's."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Protocol

from uni.template import Template

if TYPE_CHECKING:  # the model is handed in; building one is the caller's cost
    from uni.model import Model, ResidualAdd


class Knob(Protocol):
    """A scalar applied to the model. Stateless in the value, so one knob can be turned to a new value every call."""

    @property
    def spec(self) -> Mapping[str, Any] | None:
        """What the knob is, for the trajectory file; None for the knob that does nothing."""
        ...

    def additions(self, value: float) -> Sequence[ResidualAdd]: ...


class NoKnob:
    """The model unturned. A value has nothing to turn here, so only zero is a truthful one."""

    spec = None

    def additions(self, value: float) -> Sequence[ResidualAdd]:
        # [LAW:no-silent-failure] a trajectory recording a value nothing applied reads back as a steering run.
        if value:
            raise ValueError(f"there is no knob to turn, so the value must be 0, got {value}")
        return ()


@dataclass(frozen=True)
class ModelMap:
    """The pinned model under a template and a knob: the next state is the model's reply to the rendered state."""

    model: Model
    template: Template
    knob: Knob

    @property
    def spec(self) -> dict[str, Any]:
        return {
            "kind": "model",
            "template": {"name": self.template.name, "text": self.template.text},
            "pinned": asdict(self.model.pinned),
            "knob": self.knob.spec,
        }

    def step(self, state: str, value: float) -> str:
        # [LAW:dataflow-not-control-flow] every step goes through the knob; NoKnob's additions are empty.
        return self.model.generate(self.template.render(state), self.knob.additions(value)).text
