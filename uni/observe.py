"""Observables: the numbers one step of an orbit is worth plotting, read back off a written trajectory.

An observable is computed from the file and never during the run, so a new observable can be
asked of an orbit recorded months ago. PROJECT.md plots the observable at step n + 1 against
step n; the state's own text is what the period detector compares, and is not a number at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from uni.loop import Trajectory
from uni.model import Model, ModelError, ResidualAdd
from uni.parse import ConfigError, field
from uni.pinned import Pinned
from uni.steer import Direction, Steer, read_direction
from uni.template import Template, TemplateError, parse_template


class ObserveError(ConfigError):
    """A trajectory does not carry what an observable needs. The message says what is missing."""


@dataclass(frozen=True)
class Step:
    """One step of an orbit: the state it came from, and the state it reached."""

    index: int  # step 1 is the first state after the start
    previous: str
    state: str


def steps(trajectory: Trajectory) -> tuple[Step, ...]:
    """Every step of the orbit. Whatever the states mean, each one has the state it came from."""
    came_from = (trajectory.start, *trajectory.states)
    pairs = enumerate(zip(came_from, trajectory.states), start=1)
    return tuple(Step(index, previous, state) for index, (previous, state) in pairs)


class Observable(Protocol):
    """A number read off one step, for the return map."""

    @property
    def name(self) -> str: ...

    def read(self, step: Step) -> float: ...


@dataclass(frozen=True)
class Length:
    """Characters in the state. The one observable any map's states can answer."""

    @property
    def name(self) -> str:
        return "length"

    def read(self, step: Step) -> float:
        return float(len(step.state))


@dataclass(frozen=True)
class Logprob:
    """How likely the model finds the state it wrote, per token of it, at the setting it wrote it at."""

    model: Model
    template: Template
    additions: Sequence[ResidualAdd]  # what the knob added during the run; empty for an unsteered one

    @property
    def name(self) -> str:
        return "logprob"

    def read(self, step: Step) -> float:
        return self.model.reply_logprob(self.template.render(step.previous), step.state, self.additions)


@dataclass(frozen=True)
class Projection:
    """How far along a steering direction the state sits, read where the knob writes."""

    model: Model
    template: Template
    direction: Direction

    @property
    def name(self) -> str:
        return f"along:{self.direction.contrast.name}"

    def read(self, step: Step) -> float:
        # Read with the knob off, unlike the log-probability: the addition the knob makes at this
        # layer is the same vector at every step, so reading through it would add a constant that
        # says nothing about the state and everything about the setting.
        prompt = self.template.render(step.previous)
        return self.direction.project(self.model.reply_residual(prompt, step.state, self.direction.contrast.layer))


def template_of(trajectory: Trajectory) -> Template:
    """The template the trajectory recorded, so the prompt behind each step can be rendered again."""
    # [LAW:one-source-of-truth] the prompts are derived from the template the file already
    # carries; writing them beside the states would be a second copy free to disagree with it.
    kind = field(trajectory.map, "kind", str, ObserveError)
    if kind != "model":
        raise ObserveError(f"a {kind!r} trajectory has no prompts behind its states, so it has no model observables")
    raw = field(trajectory.map, "template", dict, ObserveError)
    try:
        return parse_template(field(raw, "name", str, ObserveError), field(raw, "text", str, ObserveError))
    except TemplateError as error:
        raise ObserveError(f"the template this trajectory recorded is not one: {error}") from error


def written_by(trajectory: Trajectory, pinned: Pinned) -> Pinned:
    """`pinned`, refused unless it is the checkpoint this orbit was written on."""
    # [LAW:no-silent-failure] the weights are what the numbers mean: read under any others, every
    # reading is a real number about a model that never saw this orbit. Hand the checked value on
    # rather than answering yes, so the model an observable uses is one that came through here.
    raw = field(trajectory.map, "pinned", dict, ObserveError)
    recorded = {key: field(raw, key, str, ObserveError) for key in pinned.checkpoint}
    if recorded != pinned.checkpoint:
        raise ObserveError(f"this orbit was written on {recorded}, and the pinned model is now {pinned.checkpoint}")
    return pinned


def steering_directions(trajectory: Trajectory, pinned: Pinned) -> tuple[Direction, ...]:
    """Every direction that steered this run, which is none for an unsteered one."""
    # [LAW:dataflow-not-control-flow] a collection, so a caller building observables iterates
    # rather than asking whether there is a direction at all.
    if trajectory.map.get("knob") is None:
        return ()
    knob = field(trajectory.map, "knob", dict, ObserveError)
    direction = read_direction(field(knob, "direction", str, ObserveError), pinned)
    # [LAW:no-silent-failure] projecting onto a direction that is not the one that steered the
    # run would answer the question asked, in the wrong units, and look exactly like an answer.
    if direction.sha256 != field(knob, "sha256", str, ObserveError):
        raise ObserveError(f"{direction.contrast.name} has changed since it steered this run; the trajectory records a different file")
    return (direction,)


def steering_additions(directions: Sequence[Direction], value: float) -> tuple[ResidualAdd, ...]:
    """What the knob added to the model during the run, rebuilt by the knob that added it."""
    # [LAW:one-source-of-truth] the trajectory records the direction and the value, and Steer turns
    # them into an addition exactly once; writing that arithmetic out again here is a second copy.
    return tuple(addition for direction in directions for addition in Steer(direction).turn(value).additions)


def readings(observables: Sequence[Observable], step: Step) -> tuple[float, ...]:
    """Every observable's number for one step, or a refusal that says which step has no number."""
    try:
        return tuple(observable.read(step) for observable in observables)
    except ModelError as error:  # truthful already; what it cannot know is which step it was reading
        raise ObserveError(f"step {step.index}: {error}") from error


def identities(states: Sequence[str]) -> tuple[int, ...]:
    """Each state numbered by the order it first appeared in, so an orbit's repeats line up in a column.

    Numbered rather than hashed: the detector compares the states themselves, and a number that
    counts distinct states cannot collide the way a shortened hash printed beside them could.
    """
    seen: dict[str, int] = {}
    return tuple(seen.setdefault(state, len(seen) + 1) for state in states)
