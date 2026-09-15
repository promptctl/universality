"""The maps the loop runner iterates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from uni.template import Template

if TYPE_CHECKING:  # the model is handed in; building one is the caller's cost
    from uni.model import Model


@dataclass(frozen=True)
class ModelMap:
    """The pinned model under a template: the next state is the model's reply to the rendered state."""

    model: Model
    template: Template

    @property
    def spec(self) -> dict[str, Any]:
        return {
            "kind": "model",
            "template": {"name": self.template.name, "text": self.template.text},
            "pinned": asdict(self.model.pinned),
            "knob": None,  # nothing applies the value to the model yet, so it cannot change the orbit
        }

    def step(self, state: str, value: float) -> str:
        return self.model.generate(self.template.render(state)).text
