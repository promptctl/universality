"""Observables: the numbers one step of an orbit is worth plotting, read back off a written trajectory.

An observable is computed from the file and never during the run, so a new observable can be
asked of an orbit recorded months ago. PROJECT.md plots the observable at step n + 1 against
step n; the state's own text is what the period detector compares, and is not a number at all.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

from uni.loop import Trajectory
from uni.maps import NUMBERS
from uni.model import Model, ResidualAdd
from uni.parse import ConfigError, field, nullable
from uni.pinned import Pinned, load_pins
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

    @property
    def needs(self) -> tuple[Weights, ...]:
        """The checkpoints reading this loads, so a caller can load them before it names any step or cell."""
        ...

    def read(self, step: Step) -> float: ...


@dataclass(frozen=True)
class Weights:
    """One checkpoint observables read through, held so that many orbits read it once.

    Held rather than loaded, and for the reason `ModelFamily` holds `Pinned` rather than a `Model`:
    saying what an orbit can be read for is a description, and a checkpoint is a resource.
    [LAW:carrying-cost]
    """

    pinned: Pinned

    @cached_property
    def model(self) -> Model:
        return Model(self.pinned)


class Checkpoints:
    """The checkpoints a command's orbits were written on, each loaded at most once, and only when a reading needs it.

    One of these is made per command and handed to every orbit that command reads, so a picture
    drawn from a thousand cells reads its model once and not a thousand times, and an orbit is read
    with the weights it recorded, whichever of the pinned models those are.
    """

    def __init__(self) -> None:
        self._held: dict[Pinned, Weights] = {}

    def writing(self, trajectory: Trajectory) -> Weights:
        """The weights this orbit was written on, refused unless a pinned model is those weights."""
        pinned = written_by(trajectory)
        return self._held.setdefault(pinned, Weights(pinned))


@dataclass(frozen=True)
class Length:
    """Characters in the state. The one observable any map's states can answer."""

    @property
    def name(self) -> str:
        return "length"

    @property
    def needs(self) -> tuple[Weights, ...]:
        return ()

    def read(self, step: Step) -> float:
        return float(len(step.state))


@dataclass(frozen=True)
class Value:
    """The state read as the number it is.

    For a map whose states are numbers there is nothing to derive: the state is the observable,
    and the return map PROJECT.md plots is the map itself drawn. That is what makes the logistic
    orbit the fixture - its plot is a shape that is either the published one or visibly not.
    """

    number: Callable[[str], float]  # the map's own reading of its state, which refuses a state it could not have written

    @property
    def name(self) -> str:
        return "x"

    @property
    def needs(self) -> tuple[Weights, ...]:
        return ()

    def read(self, step: Step) -> float:
        return self.number(step.state)


@dataclass(frozen=True)
class Logprob:
    """How likely the model finds the state it wrote, per token of it, at the setting it wrote it at."""

    weights: Weights  # the checkpoint this would read through, read when a reading is actually asked for
    template: Template
    additions: Sequence[ResidualAdd]  # what the knob added during the run; empty for an unsteered one

    @property
    def name(self) -> str:
        return "logprob"

    @property
    def needs(self) -> tuple[Weights, ...]:
        return (self.weights,)

    def read(self, step: Step) -> float:
        return self.weights.model.reply_logprob(self.template.render(step.previous), step.state, self.additions)


@dataclass(frozen=True)
class Projection:
    """How far along a steering direction the state sits, read where the knob writes."""

    weights: Weights  # as Logprob holds it, and for the same reason
    template: Template
    direction: Direction

    @property
    def name(self) -> str:
        return f"along:{self.direction.contrast.name}"

    @property
    def needs(self) -> tuple[Weights, ...]:
        return (self.weights,)

    def read(self, step: Step) -> float:
        # Read with the knob off, unlike the log-probability: the addition the knob makes at this
        # layer is the same vector at every step, so reading through it would add a constant that
        # says nothing about the state and everything about the setting.
        prompt = self.template.render(step.previous)
        residual = self.weights.model.reply_residual(prompt, step.state, self.direction.contrast.layer)
        return self.direction.project(residual)


def template_of(trajectory: Trajectory) -> Template:
    """The template the trajectory recorded, so the prompt behind each step can be rendered again."""
    # [LAW:one-source-of-truth] the prompts are derived from the template the file already
    # carries; writing them beside the states would be a second copy free to disagree with it.
    raw = field(trajectory.map, "template", dict, ObserveError)
    try:
        return parse_template(field(raw, "name", str, ObserveError), field(raw, "text", str, ObserveError))
    except TemplateError as error:
        raise ObserveError(f"the template this trajectory recorded is not one: {error}") from error


def written_by(trajectory: Trajectory) -> Pinned:
    """The pinned model this orbit was written on, refused unless one of them is the checkpoint it recorded."""
    # [LAW:no-silent-failure] the weights are what the numbers mean: read under any others, every
    # reading is a real number about a model that never saw this orbit. Hand the found value on
    # rather than answering yes, so the model an observable uses is one that came through here.
    raw = field(trajectory.map, "pinned", dict, ObserveError)
    recorded = {key: field(raw, key, str, ObserveError) for key in ("model_id", "revision", "dtype")}
    pins = load_pins()
    pinned = pins.writing(recorded)
    if pinned is None:
        raise ObserveError(f"this orbit was written on {recorded}, and no model pinned in uni/pinned.toml is that checkpoint; the models are {', '.join(pins.models)}")
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


@contextmanager
def addressed(address: str) -> Iterator[None]:
    """Refusals from inside, named after the one step or file they are about.

    ConfigError and not ObserveError: a trajectory is refused by `read_trajectory` for its shape,
    by `read_direction` for a direction that has changed, and by the observables for what its
    states are made of - sibling error types under the one the CLI reports. [LAW:single-enforcer]
    """
    try:
        yield
    except ConfigError as error:
        raise ObserveError(f"{address}: {error}") from error


def load_checkpoints(observables: Sequence[Observable], steps: Sequence[Step]) -> None:
    """Load every checkpoint these observables read through at these steps, before any reading can be named after a step or a cell.

    A checkpoint that cannot load is a refusal about the whole run, so it is taken here, where no
    step or cell has a name yet, and not at the first reading that happens to want one, where it
    would come out addressed to that step. Each `Weights` holds its model once loaded, so the
    readings after this load nothing. [LAW:no-silent-failure]

    Only a reading actually taken pays, so with no steps to read nothing is loaded: an orbit
    with no states, or a burn-in past a cell's last step, has no reading for a checkpoint to cost.
    [LAW:carrying-cost]
    """
    if not steps:
        return
    for observable in observables:
        for weights in observable.needs:
            weights.model


def readings(observables: Sequence[Observable], step: Step) -> tuple[float, ...]:
    """Every observable's number for one step, or a refusal that says which step has no number."""
    # The template, the directions and the checkpoint were all settled before the table started,
    # so what fails here is about this step's states.
    with addressed(f"step {step.index}"):
        return tuple(observable.read(step) for observable in observables)


def model_observables(trajectory: Trajectory, checkpoints: Checkpoints) -> tuple[Observable, ...]:
    """What a model's orbit can be read for, past the length any orbit answers.

    Reading any of these costs the checkpoint, and it is the `Weights` that own it - held by each of
    them rather than read into them, so that saying what this orbit can be read for costs nothing
    and only a reading actually taken pays. Naming a model orbit's observables, or drawing a
    picture of the one observable every map answers, therefore loads no model at all, and on a
    machine with no Metal it is an answer rather than a raise. [LAW:carrying-cost]
    """
    template = template_of(trajectory)
    # Found from this orbit even when a sweep's cells share it: the weights are what the numbers
    # mean, and a sweep's cells all recording one checkpoint is not the same fact as each of them
    # recording the one they are read with. [LAW:no-silent-failure]
    weights = checkpoints.writing(trajectory)
    directions = steering_directions(trajectory, weights.pinned)
    return (
        Logprob(weights, template, steering_additions(directions, trajectory.value)),
        *(Projection(weights, template, direction) for direction in directions),
    )


def numeric_observables(trajectory: Trajectory, checkpoints: Checkpoints) -> tuple[Observable, ...]:
    """What an orbit of numbers can be read for: the numbers, in the map's own spelling, and no checkpoint to read them."""
    return (Value(NUMBERS[trajectory.map["kind"]](trajectory.map).read),)


# What each kind of map's states can be read for, past the length every state has. A map that is
# not in here is one this build cannot read, which is a thing to say rather than to answer around.
KINDS: Mapping[str, Callable[[Trajectory, Checkpoints], tuple[Observable, ...]]] = {
    "model": model_observables,
    **{kind: numeric_observables for kind in NUMBERS},
}


def observables(trajectory: Trajectory, checkpoints: Checkpoints) -> tuple[Observable, ...]:
    """Every number this orbit can be read for, which is decided by what its states are made of.

    [LAW:dataflow-not-control-flow] the kind is looked up rather than branched on, once, here;
    what comes back is a collection, and what prints the table never asks whose orbit it is.
    """
    kind = field(trajectory.map, "kind", str, ObserveError)
    if kind not in KINDS:
        # [LAW:no-silent-failure] answering with the length alone would read as a full reading of
        # a file this build has no observables for, and the length is never the interesting one.
        raise ObserveError(f"a {kind!r} orbit is not one this build can read; the kinds are {', '.join(KINDS)}")
    return (Length(), *KINDS[kind](trajectory, checkpoints))


def identities(states: Sequence[str]) -> tuple[int, ...]:
    """Each state numbered by the order it first appeared in, so an orbit's repeats line up in a column.

    Numbered rather than hashed: the detector compares the states themselves, and a number that
    counts distinct states cannot collide the way a shortened hash printed beside them could.
    """
    seen: dict[str, int] = {}
    return tuple(seen.setdefault(state, len(seen) + 1) for state in states)
