"""Loop templates: fixed text that turns a state into the prompt, parsed once from templates.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from importlib.resources import files

from uni.parse import ConfigError

SLOT = "{state}"


class TemplateError(ConfigError):
    """templates.toml does not hold templates. The message names the file or the template."""


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
    try:
        raw = tomllib.loads(files("uni").joinpath("templates.toml").read_text())
    except tomllib.TOMLDecodeError as error:
        raise TemplateError(f"uni/templates.toml is not TOML: {error}") from error
    return {name: parse_template(name, text) for name, text in raw.items()}
