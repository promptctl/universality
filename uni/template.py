"""Loop templates: fixed text that turns a state into the prompt, parsed once from templates.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from importlib.resources import files

SLOT = "{state}"


class TemplateError(Exception):
    """A template is not text holding the slot exactly once. The message names the template."""


@dataclass(frozen=True)
class Template:
    name: str
    before: str  # the text ahead of the slot
    after: str  # the text behind it

    @property
    def text(self) -> str:
        return self.before + SLOT + self.after

    def render(self, state: str) -> str:
        # Joined, not substituted, so a state that itself contains the slot is sent as written.
        return self.before + state + self.after


def parse_template(name: str, text: object) -> Template:
    # [LAW:parse-dont-validate] one slot, found once here, so render never asks where the state goes.
    if type(text) is not str:
        raise TemplateError(f"template {name!r} must be a string, got {text!r}")
    parts = text.split(SLOT)
    if len(parts) != 2:
        raise TemplateError(f"template {name!r} must hold {SLOT} exactly once, found it {len(parts) - 1} times")
    before, after = parts
    return Template(name, before, after)


def load_templates() -> dict[str, Template]:
    raw = tomllib.loads(files("uni").joinpath("templates.toml").read_text())
    return {name: parse_template(name, text) for name, text in raw.items()}
