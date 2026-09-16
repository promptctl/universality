"""Observables: the numbers one step of an orbit is worth plotting, read back off a written trajectory.

An observable is computed from the file and never during the run, so a new observable can be
asked of an orbit recorded months ago. PROJECT.md plots the observable at step n + 1 against
step n; the state's own text is what the period detector compares, and is not a number at all.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from uni.loop import Trajectory
from uni.maps import logistic_state
from uni.model import Model, ResidualAdd
from uni.parse import ConfigError, field, nullable
from uni.pinned import Pinned, load_pinned
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
class Value:
    """The state read as the number it is.

    For a map whose states are numbers there is nothing to derive: the state is the observable,
    and the return map PROJECT.md plots is the map itself drawn. That is what makes the logistic
    orbit the fixture - its plot is a shape that is either the published one or visibly not.
    """

    @property
    def name(self) -> str:
        return "x"

    def read(self, step: Step) -> float:
        return logistic_state(step.state)


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
    # [LAW:no-silent-failure] null is how an unsteered run is recorded; a file that has lost the
    # field records nothing, and reading it as unsteered scores a steered orbit on a flat model.
    knob = nullable(trajectory.map, "knob", dict, ObserveError)
    if knob is None:
        return ()
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
    # Every observable was built before the table started printing - the checkpoint read, the
    # template parsed, the directions checked against it - so nothing about the run's description
    # is still waiting to fail here, and a refusal that arrives is about this one state. It is
    # truthful already; what it cannot know is which step it was reading.
    except ConfigError as error:
        raise ObserveError(f"step {step.index}: {error}") from error


def model_observables(trajectory: Trajectory) -> tuple[Observable, ...]:
    """What a model's orbit can be read for, past the length any orbit answers.

    Reading any of these costs the checkpoint, so it is loaded here and only here: an orbit of a
    map that has no prompts behind its states never pays for one.
    """
    template = template_of(trajectory)
    pinned = written_by(trajectory, load_pinned())
    directions = steering_directions(trajectory, pinned)
    model = Model(pinned)
    return (
        Logprob(model, template, steering_additions(directions, trajectory.value)),
        *(Projection(model, template, direction) for direction in directions),
    )


def numeric_observables(trajectory: Trajectory) -> tuple[Observable, ...]:
    """What an orbit of numbers can be read for: the numbers."""
    return (Value(),)


# What each kind of map's states can be read for, past the length every state has. A map that is
# not in here is one this build cannot read, which is a thing to say rather than to answer around.
KINDS: Mapping[str, Callable[[Trajectory], tuple[Observable, ...]]] = {
    "model": model_observables,
    "logistic": numeric_observables,
}


def observables(trajectory: Trajectory) -> tuple[Observable, ...]:
    """Every number this orbit can be read for, which is decided by what its states are made of.

    [LAW:dataflow-not-control-flow] the kind is looked up rather than branched on, once, here;
    what comes back is a collection, and what prints the table never asks whose orbit it is.
    """
    kind = field(trajectory.map, "kind", str, ObserveError)
    if kind not in KINDS:
        # [LAW:no-silent-failure] answering with the length alone would read as a full reading of
        # a file this build has no observables for, and the length is never the interesting one.
        raise ObserveError(f"a {kind!r} orbit is not one this build can read; the kinds are {', '.join(KINDS)}")
    return (Length(), *KINDS[kind](trajectory))


def identities(states: Sequence[str]) -> tuple[int, ...]:
    """Each state numbered by the order it first appeared in, so an orbit's repeats line up in a column.

    Numbered rather than hashed: the detector compares the states themselves, and a number that
    counts distinct states cannot collide the way a shortened hash printed beside them could.
    """
    seen: dict[str, int] = {}
    return tuple(seen.setdefault(state, len(seen) + 1) for state in states)
