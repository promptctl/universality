"""The maps the loop runner iterates, and the knobs that turn the model's."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from functools import cached_property, partial
from typing import TYPE_CHECKING, Any, Protocol

from uni.parse import ConfigError
from uni.template import Template

if TYPE_CHECKING:  # a family reads its own checkpoint; everything else here is handed what it holds
    from uni.loop import Map
    from uni.model import Model, ResidualAdd
    from uni.pinned import Pinned
    from uni.smooth import Series
    from uni.steer import Steer


class MapError(ConfigError):
    """A map cannot be built as described, or cannot step the state it was handed."""


class KnobError(ConfigError):
    """A knob was asked for a value it cannot be turned to."""


@dataclass(frozen=True)
class Turned:
    """A knob at one setting: the setting, what it adds to the model, and what it was, for the file."""

    value: float  # what it was turned to, which is the number the trajectory records
    spec: Mapping[str, Any] | None  # None for the knob that does nothing
    additions: Sequence[ResidualAdd]


class Knob(Protocol):
    """A scalar the model can be turned by. Turning is by value, so one knob serves a whole sweep."""

    @property
    def spec(self) -> Mapping[str, Any] | None:
        """What this knob is. Its setting does not change it, so a sweep can name it unturned."""
        ...

    def turn(self, value: float) -> Turned: ...


class NoKnob:
    """The model unturned. A value has nothing to turn here, so only zero is a truthful one."""

    spec = None  # a recorded nothing, which a reader tells apart from a file that lost the field

    def turn(self, value: float) -> Turned:
        # [LAW:no-silent-failure] a trajectory recording a value nothing applied reads back as a steering run.
        if value:
            raise KnobError(f"there is no knob to turn, so the value must be 0, got {value}")
        return Turned(value, self.spec, ())


class Family(Protocol):
    """A map with everything fixed but its parameter: what a sweep turns, and what one run turns once.

    The knob interface already says a setting is per call rather than per run; this says the same
    thing one level up, about the whole map. A sweep holds one family and asks it for a map at
    each value on the grid, so the checkpoint behind a model sweep is loaded once and not once a
    cell. [LAW:composability]
    """

    @property
    def spec(self) -> Mapping[str, Any]:
        """What every map in this family is, which the parameter does not change.

        Asked of the family and not of a map at some one value off the grid, because it is the
        family's own answer: a sweep is named by this before it runs anything, and a family that
        had to make a map to say what it is would load a checkpoint to answer `--status`.
        [LAW:one-source-of-truth]
        """
        ...

    def holds(self, states: Sequence[str]) -> None:
        """Refuse any of these states that no map of this family could step.

        The parameter does not move the states a map holds - r does not change [0, 1], and a knob
        setting does not move where the context limit falls - so which states are legal is the
        family's own answer. It is asked once a run is about to begin, not when a sweep is merely
        being named, so a family may use whatever it would load to run anyway: the answer costs
        nothing a cell was not about to cost. A sweep handed a start its map cannot step would
        otherwise write its manifest, run every cell before that start and die there - and die
        again in the same place on every resume. [LAW:no-silent-failure]
        """
        ...

    def at(self, value: float) -> Map:
        """This map with its parameter set, at exactly that value: `at(v).value == v`."""
        ...


def model_spec(pinned: Pinned, template: Template, knob: Mapping[str, Any] | None) -> dict[str, Any]:
    """What a model map is, in the one form a trajectory and a sweep manifest both record.

    Written here and read by both a family and the maps it makes, so a sweep cannot describe one
    map in its manifest and another in its cells. [LAW:one-source-of-truth]
    """
    return {
        "kind": "model",
        "template": {"name": template.name, "text": template.text},
        "pinned": asdict(pinned),
        "knob": knob,
        # What becomes of a reply the token budget cut off, recorded because it is part of what the
        # map is: orbits written before it was refused hold those cuts as states, and without this
        # they would be named, and resumed, as orbits of this map. [LAW:one-source-of-truth]
        "truncated": "refused",
    }


@dataclass(frozen=True)
class ModelMap:
    """The pinned model under a template and a turned knob: the next state is the model's reply to the rendered state."""

    model: Model
    template: Template
    knob: Turned

    @property
    def value(self) -> float:
        # The knob's setting is this map's one parameter, and it is held once, by the knob that
        # was turned to it. [LAW:one-source-of-truth]
        return self.knob.value

    @property
    def spec(self) -> dict[str, Any]:
        return model_spec(self.model.pinned, self.template, self.knob.spec)

    # What cut a reply off, named so that whoever fixes the refused cell turns the thing that cut it.
    CUT_BY = {
        "budget": "the pinned generation.max_new_tokens of {tokens}",
        "context": "the {tokens} tokens the context limit leaves after this prompt",
    }

    def step(self, state: str) -> str:
        # [LAW:dataflow-not-control-flow] every step adds the same additions; the unturned knob's are empty.
        generation = self.model.generate(self.template.render(state), self.knob.additions)
        # The next state is the model's reply, and a reply a limit cut off is not one: it is the
        # limit's cut of a reply whose end was never generated, and an orbit of those is an orbit
        # of the cap. Steered far enough, the model repeats itself where no budget would end it, so
        # the step is refused rather than recorded - in a sweep, a refused cell. [LAW:no-silent-failure]
        if generation.ended != "stop":
            cut_by = self.CUT_BY[generation.ended].format(tokens=len(generation.token_ids))
            raise MapError(f"the reply did not end within {cut_by}, so it is a cut of a state rather than one")
        return generation.text


@dataclass(frozen=True)
class ModelFamily:
    """The pinned configuration under a template, with the knob described but not yet turned.

    It holds what the model *is* rather than a loaded one, because a family is a description and a
    checkpoint is a resource. `uni sweep --status` names a sweep, counts the files it already has
    and answers; half a billion parameters are no part of that question, and on a machine with no
    Metal loading them is not a slow answer but no answer at all. The checkpoint is read by the
    first `at`, which is the first time anything asks for a map to actually run.
    """

    pinned: Pinned
    template: Template
    knob: Knob

    @property
    def spec(self) -> dict[str, Any]:
        return model_spec(self.pinned, self.template, self.knob.spec)

    def holds(self, states: Sequence[str]) -> None:
        # A model map steps any text the context leaves room to answer, which is the model's own
        # arithmetic rather than a second copy of it here. The checkpoint this asks is the one the
        # run is about to load, so a start that cannot be stepped costs the same to find out about
        # now as it would have cost a cell in - and a cell in, the manifest is already written and
        # every resume dies in the same place.
        for state in states:
            self.model.room(self.model.encode(self.template.render(state)))

    @cached_property
    def model(self) -> Model:
        # Held once, so a sweep of a thousand cells reads the checkpoint once and not once a cell.
        # Imported here rather than at the top: naming a model map must not cost torch.
        from uni.model import Model

        return Model(self.pinned)

    def at(self, value: float) -> Map:
        # Imported here for the reason `model` above gives, and at no extra cost: the checkpoint
        # this reaches for on the next line has already paid for torch.
        from uni.model import ResidualAdd

        turned = self.knob.turn(value)
        # [LAW:parse-dont-validate] the additions become ones this checkpoint can take, or no map
        # is made. What an addition adds to is fixed by the direction and not by the setting, so a
        # layer this checkpoint does not have, or a vector of the wrong length, is wrong in every
        # cell of a sweep - which is why it is caught here, before a token is generated, rather
        # than once per cell by a run that can never write a file. The vector reaches the device
        # once per map rather than once per step as well.
        additions = tuple(ResidualAdd(one.layer, self.model.residual_vector(one)) for one in turned.additions)
        return ModelMap(self.model, self.template, replace(turned, additions=additions))


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
    # [LAW:one-source-of-truth] one number, one text. The detector compares the state's text and
    # this reads its number, so a second spelling of one number - a `--start 0.50` - is one state
    # to the plot and two to the verdict: a fixed point the period column would deny having.
    if repr(x) != state:
        raise MapError(f"a logistic state is the shortest text that reads back as its number: write {repr(x)}, not {state!r}")
    return x


# r is not in here: it is the trajectory's value, which the runner reads off `value` below, in the
# one place a steering setting is written too. Written once for the family and for the maps it
# makes, which have to say the same thing. [LAW:one-source-of-truth]
LOGISTIC = {"kind": "logistic"}


class LogisticFamily:
    """x -> r x (1 - x) with r still to come. Nothing to hold: r is the whole of this map."""

    @property
    def spec(self) -> dict[str, Any]:
        return dict(LOGISTIC)

    def holds(self, states: Sequence[str]) -> None:
        # The function `step` reads a state with, so there is no second rule about which states
        # this map holds, free to drift from the one that decides. [LAW:one-source-of-truth]
        for state in states:
            logistic_state(state)

    def at(self, value: float) -> Map:
        return Logistic(value)


@dataclass(frozen=True)
class Logistic:
    """x -> r x (1 - x) on [0, 1]: the map whose cascade every other map here is measured against.

    Textbook, and that is the point. Feigenbaum's period doubling is known for this map to more
    decimal places than this pipeline will ever resolve, so an orbit of it is the one fixture
    where a wrong answer is visible as a wrong answer rather than as a result.
    """

    r: float

    @property
    def value(self) -> float:
        """r under the protocol's name for it: the one number besides the start that fixes the orbit."""
        return self.r

    def __post_init__(self) -> None:
        # [LAW:parse-dont-validate] past this line a Logistic exists, and one exists only at an r
        # that keeps [0, 1] inside itself. Outside, the orbit runs to -inf and then to nan, and a
        # run of nans is a state that repeats: the detector would report period 1 on a divergence.
        if not 0 <= self.r <= 4:
            raise MapError(f"r must be in 0..4, or the map carries [0, 1] out of itself and the orbit diverges; got {self.r}")

    @property
    def spec(self) -> dict[str, Any]:
        return dict(LOGISTIC)

    def slope(self, x: float) -> float:
        return self.r * (1 - 2 * x)

    def step(self, state: str) -> str:
        x = logistic_state(state)
        # repr, which is the shortest text that reads back as exactly this float. The detector
        # compares the states themselves, so two spellings of one number would be two states.
        return repr(self.r * x * (1 - x))


# The decimals a response state is written to unless a run asks for others. The model's answer is
# float32, and a push rounds to float32 on its way into the stream, so a push held to every bit of a
# float64 wanders among neighbours a millionth apart that the model cannot tell from one another:
# measured at gains 1 to 12, a settled orbit's states spread over 2e-7 to 7e-6, which the exact
# detector reads as a cycle of three, or six, or as a real period doubled. Four decimals is fourteen
# times the widest of those, and fine enough for an orbit diagram. What detects no period can read
# more finely: `uni cascade` at six decimals sees its returns scatter three to thirty-five times less,
# and what scatter is left is the model's own float32 roughness, which more decimals would not remove.
RESPONSE_DECIMALS = 4


def response_text(push: float, decimals: int) -> str:
    """A push as the one text the response map writes for it. Rounded before it is written, so -0.00001 is 0.0000 and not -0.0000."""
    return f"{round(push, decimals) + 0.0:.{decimals}f}"


def response_state(state: str, decimals: int) -> float:
    """The push a response state names, refused unless it is written as the map writes one.

    One function for the map and the observable, as `logistic_state` is: the detector compares the
    state's text and the plot reads its number, so a second spelling of a push would be one point
    to the plot and two states to the verdict. [LAW:one-source-of-truth]
    """
    try:
        push = float(state)
    except ValueError as error:
        raise MapError(f"a response state is a push written to {decimals} decimals, got {state!r}") from error
    if not math.isfinite(push) or response_text(push, decimals) != state:
        written = response_text(push, decimals) if math.isfinite(push) else f"a number to {decimals} decimals"
        raise MapError(f"a response state is a push written to {decimals} decimals: write {written}, not {state!r}")
    return push


def response_spec(pinned: Pinned, template: Template, text: str, steer: Mapping[str, Any], layer: int, decimals: int) -> dict[str, Any]:
    """What a response map is, in the one form a trajectory and a sweep manifest both record. The gain is the value, recorded beside it."""
    return {
        "kind": "response",
        "template": {"name": template.name, "text": template.text},
        "pinned": asdict(pinned),
        "text": text,
        "knob": steer,
        "layer": layer,
        "decimals": decimals,
    }


@dataclass(frozen=True)
class ResponseFamily:
    """The push-response loop with its gain still to come: everything `uni response` reads a curve with, fixed.

    The prompt is rendered once from `text`, and the state is not text but the push. Held as the
    pinned configuration rather than a loaded model, for the reason `ModelFamily` gives.
    """

    pinned: Pinned
    template: Template
    text: str
    steer: Steer
    layer: int
    decimals: int

    @property
    def spec(self) -> dict[str, Any]:
        return response_spec(self.pinned, self.template, self.text, self.steer.spec, self.layer, self.decimals)

    @cached_property
    def model(self) -> Model:
        from uni.model import Model

        return Model(self.pinned)

    @property
    def prompt(self) -> str:
        return self.template.render(self.text)

    def holds(self, states: Sequence[str]) -> None:
        # [LAW:single-enforcer] a start is a push the response reads, so it is refused by what
        # refuses a push: the spelling, and then the reading's own rounding bound.
        from uni.response import admit

        for state in states:
            admit(self.model, self.steer, response_state(state, self.decimals), self.layer)

    def at(self, value: float) -> Map:
        from uni.response import admit

        # What the whole family reads with - the layer, the prompt's length - is refused here, at
        # no push at all, so a sweep that can never read a cell is refused before its first one.
        admit(self.model, self.steer, 0.0, self.layer)
        self.model.fits(self.model.encode(self.prompt), "the prompt's tokens")
        return ResponseMap(self, value)


@dataclass(frozen=True)
class ResponseMap:
    """x -> gain * r(x) / |v|^2: the model's answer to a push, turned back into the push that would read as it, times the gain.

    Divided by the squared length because a push of x reads back as x |v|^2 along the direction, so
    r / |v|^2 is the answer in the units of the push; the gain is then the one number that says how
    much of the answer is fed back, which is PROJECT.md's feedback gain made literal. At gain 1 the
    next push is exactly the push the answer amounts to.
    """

    family: ResponseFamily
    gain: float

    @property
    def value(self) -> float:
        return self.gain

    @property
    def spec(self) -> dict[str, Any]:
        return self.family.spec

    def step(self, state: str) -> str:
        from uni.response import response

        family = self.family
        answer = response(family.model, family.prompt, family.steer, response_state(state, family.decimals), family.layer)
        return response_text(self.gain * answer / family.steer.direction.squared_length, family.decimals)


def smooth_state(state: str) -> float:
    """The push a smooth map's state names: a float, spelled as its own shortest text, as a logistic state is."""
    try:
        push = float(state)
    except ValueError as error:
        raise MapError(f"a smooth map's state is a push, a number, got {state!r}") from error
    if not math.isfinite(push):
        raise MapError(f"a smooth map's state is a finite push, got {state!r}")
    # Every bit kept, where the response map keeps four or six decimals: the series is float64 all
    # the way through and has no roughness for the bits below that to be lost in.
    if repr(push) != state:
        raise MapError(f"a smooth map's state is the shortest text that reads back as its push: write {repr(push)}, not {state!r}")
    return push


@dataclass(frozen=True)
class SmoothFamily:
    """x -> gain * C(x) / |v|^2, where C is a series fitted to the response map's answers, with the gain still to come.

    The response map with the model replaced by what the model's curve is, fitted smooth: the same
    division by the squared length, so a gain here is the gain there.
    """

    curve: str  # the name of the curve fitted, which is its content
    layer: int
    degree: int
    series: Series
    squared_length: float

    @property
    def spec(self) -> dict[str, Any]:
        # [LAW:one-source-of-truth] the coefficients are the fit's to recompute from the curve and the
        # degree, so they are named and not copied. Named, because a least-squares fit need not land on
        # the same last bits under another numpy build, and a cascade to period 8192 turns on those:
        # a sweep resumed there would match a spec of the curve and degree alone and run another map.
        return {"kind": "smooth", "curve": self.curve, "layer": self.layer, "degree": self.degree, "series": self.series.name}

    def admit(self, push: float) -> None:
        """Refuse a push outside the curve's samples. [LAW:single-enforcer] a start and every step are refused by this."""
        # [LAW:no-silent-failure] past the last push it was fitted to, a series grows as its highest
        # power does, and an orbit that wandered there would be a cascade of that and not the model's.
        if not self.series.low <= push <= self.series.high:
            raise MapError(f"a push of {push!r} lies outside the curve's pushes, {self.series.low:g} to {self.series.high:g}, and a series says nothing past the readings it was fitted to")

    def holds(self, states: Sequence[str]) -> None:
        for state in states:
            self.admit(smooth_state(state))

    def at(self, value: float) -> Map:
        return SmoothMap(self, value)


@dataclass(frozen=True)
class SmoothMap:
    """x -> gain * C(x) / |v|^2 at one gain."""

    family: SmoothFamily
    gain: float

    @property
    def value(self) -> float:
        return self.gain

    @property
    def spec(self) -> dict[str, Any]:
        return self.family.spec

    def slope(self, x: float) -> float:
        return self.gain * self.family.series.slope.at(x) / self.family.squared_length

    def step(self, state: str) -> str:
        push = smooth_state(state)
        self.family.admit(push)
        return repr(self.gain * self.family.series.at(push) / self.family.squared_length)


@dataclass(frozen=True)
class Numbers:
    """How a map whose states are numbers reads a state as its number, and writes a number as the state it would be."""

    read: Callable[[str], float]
    write: Callable[[float], str]


def response_numbers(spec: Mapping[str, Any]) -> Numbers:
    """A response map's spelling of its pushes, to the decimals its spec records."""
    decimals = spec.get("decimals")
    # [LAW:parse-dont-validate] read off a file as often as off a family, and a decimals that is
    # missing, or not a positive whole number, spells no push at all.
    if type(decimals) is not int or decimals < 1:
        recorded = repr(decimals) if "decimals" in spec else "none"
        raise MapError(f"a response map's decimals are a positive whole number, and this one records {recorded}")
    return Numbers(partial(response_state, decimals=decimals), partial(response_text, decimals=decimals))


# The kinds of map whose states are numbers, and how each spells them, from what its spec records.
# What reads an orbit for its numbers and what asks a map where it holds still both look the kind
# up here, so a third numeric map is one entry and not a branch in each of them. [LAW:one-source-of-truth]
NUMBERS: Mapping[str, Callable[[Mapping[str, Any]], Numbers]] = {
    "logistic": lambda spec: Numbers(logistic_state, repr),
    "response": response_numbers,
    "smooth": lambda spec: Numbers(smooth_state, repr),
}
