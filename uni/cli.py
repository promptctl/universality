"""The `uni` command line. Every subcommand accepts --remote."""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import platform
import signal
import subprocess
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, assert_never

from dotenv import dotenv_values

from uni.determinism import RUNS
from uni.parse import ConfigError
from uni.period import Contradiction, Cycle, NoCycle, Period
from uni.remote import RemoteConfigError, remote_target_from_env, run_remote
from uni.template import Template, TemplateError, load_templates

if TYPE_CHECKING:
    from uni.maps import Family, Knob, Numbers
    from uni.steer import Contrast

EXIT_CONFIG = os.EX_CONFIG  # distinct from argparse's 2 and from anything rsync or ssh returns
EXIT_DIVERGED = 1  # the determinism gate ran and some case produced more than one hash
# The sweep ran and some cell of it has no orbit: a different answer from EXIT_CONFIG, which says
# the run as described cannot be run, because here it was run and most of it is on disk. One past
# the end of the sysexits table EXIT_CONFIG comes from, so it is no more rsync's or ssh's than
# that one is. [LAW:no-silent-failure] the distinction the determinism gate already draws with
# EXIT_DIVERGED: the command worked, and what it found is the bad news.
EXIT_INCOMPLETE = 79
# The run was fine and the machine would not do it: a full disk, a read-only volume, a sweeps
# directory mangled by hand. Distinct from EXIT_CONFIG because what the reader does about it is
# different - nothing about the command or the files it named needs changing, and a resume loop
# that would retry a fixed grid must stop for a disk that will not empty itself. The name is
# sysexits' own for this, as EXIT_CONFIG is. Until now it was the traceback's exit 1, which is
# EXIT_DIVERGED: a crash and a determinism failure read alike to anything reading the code.
EXIT_IO = os.EX_IOERR
# Nothing failed: something downstream stopped reading, which `uni sweep ... | head` does on
# purpose. 128 + the signal, which is what a shell reports for a producer killed by this same
# event - so `PIPESTATUS[0]` says the same thing whether Python handled it or died of it.
EXIT_PIPE = 128 + signal.SIGPIPE
# A bug here: an exception nothing in this program answers for, reported with its traceback. Its
# own code rather than the interpreter's 1, which is EXIT_DIVERGED, so a crash partway through a
# run cannot read as the determinism gate's verdict. sysexits' name for it, as with the others.
EXIT_BUG = os.EX_SOFTWARE

# --remote is parsed here, once, and never reaches a subcommand: what is left over is
# exactly what the host runs. [LAW:one-source-of-truth]
REMOTE = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
REMOTE.add_argument(
    "--remote",
    action="store_true",
    help="sync this working tree to the run host and run the command there",
)


def split_remote(argv: Sequence[str]) -> tuple[bool, list[str]]:
    args, rest = REMOTE.parse_known_args(argv)
    return args.remote, rest


def run_host(args: argparse.Namespace) -> int:
    """Print where this process is running, so a --remote run is visibly remote."""
    print(f"{platform.node()} {platform.machine()} {platform.system()} python {platform.python_version()}")
    return 0


def run_gen(args: argparse.Namespace) -> int:
    """Print the generated text, its hash, and each token with its log-probability."""
    # Imported here so host, --remote, and --help never pay for loading torch.
    from uni.model import Model
    from uni.pinned import load_pinned

    generation = Model(load_pinned()).generate(args.prompt)
    print(generation.text)
    print()
    print(f"sha256 {generation.sha256}")
    print(f"{'step':>4} {'logprob':>12}  token")
    for step, (token, logprob) in enumerate(zip(generation.tokens, generation.logprobs)):
        print(f"{step:>4} {logprob:>12.6f}  {token!r}")
    return 0


def whole(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must not be negative, got {value}")
    return value


def positive(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return value


def finite(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):  # nan poisons every hidden state and is not JSON
        raise argparse.ArgumentTypeError(f"must be a finite number, got {text}")
    return value


def run_determinism(args: argparse.Namespace) -> int:
    """Generate each gate case `runs` times and print every hash; exit 1 on any mismatch."""
    from uni.determinism import cases, hashes
    from uni.model import Model
    from uni.pinned import load_pinned

    model = Model(load_pinned())
    distinct = {}
    for case in cases(model):
        print(f"{case.name}: {args.runs} runs")
        print(f"{'run':>4}  sha256")
        seen = set()
        for run, sha in enumerate(hashes(model, case, args.runs), start=1):
            print(f"{run:>4}  {sha}", flush=True)  # a --remote run streams through a pipe
            seen.add(sha)
        distinct[case.name] = len(seen)
        print()
    for name, count in distinct.items():
        print(f"{name}: {'deterministic' if count == 1 else f'DIVERGED, {count} distinct hashes'}")
    return EXIT_DIVERGED if any(count != 1 for count in distinct.values()) else 0


TRAJECTORIES = Path("trajectories")  # under the directory uni runs in; the --remote sync excludes it, so the host keeps its own


def contrast(name: str) -> Contrast:
    from uni.steer import load_contrast

    try:
        return load_contrast(name)
    except ConfigError as error:  # argparse prints a traceback for anything but its own error type
        raise argparse.ArgumentTypeError(str(error)) from error


def run_direction(args: argparse.Namespace) -> int:
    """Derive a steering direction from its contrast and write it beside the contrast."""
    from uni.model import Model
    from uni.pinned import load_pinned
    from uni.steer import derive, write_direction

    direction = derive(Model(load_pinned()), args.contrast)
    path = write_direction(direction)
    print(f"{path}  layer {args.contrast.layer}  length {math.hypot(*direction.vector):.6f}  sha256 {direction.sha256}")
    return 0


def template(name: str) -> Template:
    """The template a name stands for, refused unless uni/templates.toml holds one by that name."""
    templates = load_templates()
    if name not in templates:
        raise TemplateError(f"no template {name!r}; the templates are {', '.join(templates)}")
    return templates[name]


def turned_at(knob: Knob, values: Sequence[float]) -> Knob:
    """`knob`, refused unless it takes every value it is about to be asked for.

    Checked by the same call that will later turn it for real, so there is no second rule about
    which values are allowed, free to drift from the one that matters. [LAW:one-source-of-truth]
    """
    for value in values:
        knob.turn(value)
    return knob


# The flags that describe a map, as opposed to a run of one. Declared here rather than on `uni
# loop` directly so the set has a single spelling: each map's builder names the ones it reads and
# `refuse_unread` refuses the rest, so a flag added here is refused by every map that does not
# name it without anything else being edited. A flag a map ignores is one a run would accept and
# quietly drop, which is the failure the refusal exists to prevent. [LAW:one-source-of-truth]
MAP_FLAGS = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
MAP_FLAGS.add_argument("--template", help="the model or response map's template, named in uni/templates.toml")
MAP_FLAGS.add_argument("--knob", help="a direction in uni/directions: what the model map steers along (or none), or what the response map pushes along")
MAP_FLAGS.add_argument("--text", help="the response map's text, rendered once into its template")
MAP_FLAGS.add_argument("--layer", type=whole, help="the layer the response map reads its answer at, or the smooth map's curve was read at")
MAP_FLAGS.add_argument("--decimals", type=positive, help="the decimals the response map writes a push to (default: 4)")
MAP_FLAGS.add_argument("--curve", type=Path, help="the curve the smooth map is fitted to, a file `uni response` wrote")
MAP_FLAGS.add_argument("--degree", type=positive, help="the degree of the series the smooth map fits to its curve")
MAP_OPTIONS = tuple(vars(MAP_FLAGS.parse_args([])))  # the flags above, under the names args carries them by


def refuse_unread(args: argparse.Namespace, name: str, reads: Sequence[str]) -> None:
    """Refuse every map flag given that the `name` map does not read. [LAW:single-enforcer]

    [LAW:no-silent-failure] carried over from an earlier command and dropped without a word, a flag
    would read back as a setting this run had honoured.
    """
    from uni.maps import MapError

    for flag in MAP_OPTIONS:
        if getattr(args, flag) is not None and flag not in reads:
            described = ", ".join(f"--{one}" for one in reads) or "no flag; its one parameter is the value"
            raise MapError(f"--{flag} does not describe the {name} map, which reads {described}")


def model_map(args: argparse.Namespace, values: Sequence[float]) -> Family:
    """The pinned model reading its prompt from a template, turned by the knob.

    [LAW:single-enforcer] every flag describing this map is read here, and only here. argparse
    cannot know which flags belong to the --map it was handed, so a converter on one of them
    would read a direction file, and load torch to hold it, for a run about to be refused.
    """
    from uni.maps import MapError, ModelFamily, NoKnob  # torch-free, so a refusal below costs nothing

    refuse_unread(args, "model", ("template", "knob"))
    # [LAW:types-are-the-program] exception: argparse can require a flag for neither --map or for
    # both, so the map that reads a template is the one that refuses a run without it.
    if args.template is None:
        raise MapError("the model map reads its prompt from a template; pass --template")
    prompt = template(args.template)  # a name refused here is refused before a checkpoint is read
    from uni.pinned import load_pinned  # torch-free: pinned.toml is a file, not a model

    # The knob is described here and turned per value later: a sweep turns it at every point on
    # its grid, and one `uni loop` is that same turning done exactly once.
    #
    # The two knobs differ in what they need to exist, which is the whole of this branch. The one
    # that adds nothing needs nothing, so the values it cannot take are refused before even
    # pinned.toml is read: `--value 2` with no `--knob` is a run that cannot run, and saying so
    # should cost nothing at all. A direction, by contrast, has to be checked against the
    # checkpoint it will be added to, so that branch reads the pinned config first, and it is the
    # only one that pays torch to hold a vector.
    if args.knob in (None, "none"):
        knob = turned_at(NoKnob(), values)
        pinned = load_pinned()
    else:
        pinned = load_pinned()
        from uni.steer import Steer, read_direction

        knob = turned_at(Steer(read_direction(args.knob, pinned)), values)
    # The pinned configuration and not a loaded model: the family reads the checkpoint when it is
    # first asked for a map to run, so describing this sweep costs nothing.
    return ModelFamily(pinned, prompt, knob)


def logistic_map(args: argparse.Namespace, values: Sequence[float]) -> Family:
    """x -> r x (1 - x), where the value is r. Pure arithmetic: this map never loads a checkpoint."""
    from uni.maps import LogisticFamily

    refuse_unread(args, "logistic", ())
    family = LogisticFamily()
    # [LAW:no-silent-failure] every value on the grid is offered to the map up front, as every
    # value is offered to the knob above. A sweep to r = 4.2 would otherwise write its manifest,
    # grind through a hundred and seventy cells and die on the first r the map refuses - and every
    # resume would march to the same wall again, against a total it can never reach.
    for value in values:
        family.at(value)
    return family


def response_map(args: argparse.Namespace, values: Sequence[float]) -> Family:
    """The model's answer to a push, fed back as the next push times the gain, where the value is the gain."""
    from uni.maps import RESPONSE_DECIMALS, MapError, ResponseFamily

    refuse_unread(args, "response", ("template", "knob", "text", "layer", "decimals"))
    # [LAW:types-are-the-program] exception: argparse cannot require a flag for one --map only.
    missing = [f"--{flag}" for flag in ("template", "knob", "text", "layer") if getattr(args, flag) is None]
    if missing or args.knob == "none":
        raise MapError(f"the response map pushes along a direction and reads the answer to a text at a layer; pass {', '.join(missing) or '--knob with a direction'}")
    prompt = template(args.template)
    from uni.pinned import load_pinned
    from uni.steer import Steer, read_direction

    pinned = load_pinned()
    # Every gain is a map: a negative one feeds the answer back reversed, and 0 sends every push to
    # 0. What a gain can carry the orbit into is refused where it happens, by the reading's bound.
    decimals = RESPONSE_DECIMALS if args.decimals is None else args.decimals
    return ResponseFamily(pinned, prompt, args.text, Steer(read_direction(args.knob, pinned)), args.layer, decimals)


def smooth_map(args: argparse.Namespace, values: Sequence[float]) -> Family:
    """The response map with the model's answer replaced by a series fitted to a curve of it: no checkpoint, no roughness."""
    from uni.curve import read_curve
    from uni.maps import MapError, SmoothFamily
    from uni.smooth import smooth

    refuse_unread(args, "smooth", ("curve", "layer", "degree"))
    # [LAW:types-are-the-program] exception: argparse cannot require a flag for one --map only.
    missing = [f"--{flag}" for flag in ("curve", "layer", "degree") if getattr(args, flag) is None]
    if missing:
        raise MapError(f"the smooth map is a series of some degree fitted to a curve's readings at a layer; pass {', '.join(missing)}")
    curve = read_curve(args.curve, args.layer)
    return SmoothFamily(curve.name, curve.layer, args.degree, smooth(curve, args.degree).series, curve.squared_length)


def model_value(given: float | None) -> float:
    """Unturned is a setting like any other, and the one a run that named no value means: at 0 a
    knob adds nothing, so the orbit is exactly the unsteered one."""
    return 0.0 if given is None else given


def required(parameter: str) -> Callable[[float | None], float]:
    """A --value with no default worth having, refused when it is missing with `parameter` said.

    r has none, and neither has a gain: 0 is a legal value of each, so a forgotten --value would not
    fail. It would run the map that sends every state to zero and be answered `period 1` - a period
    claim, which is this project's whole output, about a parameter nobody chose.
    """

    def value(given: float | None) -> float:
        from uni.maps import MapError

        if given is None:
            raise MapError(f"{parameter}; pass --value")
        return given

    return value


@dataclass(frozen=True)
class Kind:
    """One map as the command line knows it: how to build it, and what a missing --value means to it.

    Both halves vary by map and neither is the runner's business, so they travel together as one
    value that `--map` carries, the way main() carries the subcommand it was handed.
    """

    # A builder is handed every value the run will use, and returns a family only if it takes all
    # of them: a grid is refused whole, before anything is written, rather than partway through.
    build: Callable[[argparse.Namespace, Sequence[float]], Family]
    value: Callable[[float | None], float]  # a single run's parameter; a sweep's come from the grid


MAPS = {
    "model": Kind(model_map, model_value),
    "logistic": Kind(logistic_map, required("the logistic map's parameter is r")),
    "response": Kind(response_map, required("the response map's parameter is the gain")),
    "smooth": Kind(smooth_map, required("the smooth map's parameter is the gain")),
}


def map_named(name: str) -> Kind:
    """The map to iterate, as the pair of functions that make one from the rest of the flags.

    A value and not a branch, so nothing in the runner, the sweep or the detector knows there is
    more than one map.
    """
    if name not in MAPS:
        raise argparse.ArgumentTypeError(f"no map {name!r}; the maps are {', '.join(MAPS)}")
    return MAPS[name]


def run_loop(args: argparse.Namespace) -> int:
    """Print the start and every state as it lands, then write the trajectory file."""
    from uni.loop import Trajectory, orbit, write_trajectory

    # [LAW:parse-dont-validate] the value is settled and the map built at it before a state is
    # stepped: past this line a map exists, and a map exists only at a value it accepted.
    value = args.map.value(args.value)
    family = args.map.build(args, (value,))
    family.holds((args.start,))
    map = family.at(value)
    print(f"{'step':>4}  state")
    print(f"{0:>4}  {args.start!r}")
    states = []
    for step, state in enumerate(islice(orbit(map, args.start), args.steps), start=1):
        print(f"{step:>4}  {state!r}", flush=True)  # repr, so each state is one line and an empty one shows
        states.append(state)
    path = write_trajectory(Trajectory(map.spec, map.value, args.start, tuple(states)), TRAJECTORIES)
    print()
    print(f"trajectory {path}")
    return 0


SWEEPS = Path("sweeps")  # under the directory uni runs in; the --remote sync excludes it, so the host keeps its own


def run_sweep(args: argparse.Namespace) -> int:
    """Run every cell the sweep has not already written, and say how far it got."""
    from uni.loop import Trajectory, write_trajectory
    from uni.starts import named_starts
    from uni.sweep import Failed, Sweep, described, grid, pending, run_cell, write_sweep

    # Read here and not by argparse: a grid that is not one is the user's typo, and this repo
    # reports that as `uni: ...` with EX_CONFIG. argparse catches only its own error type, so a
    # converter raising SweepError would reach the user as a traceback, which means a bug here.
    values = grid(args.grid)
    # Built once for the whole grid, which is the point of a family: the checkpoint behind a model
    # sweep is read once, and not until a cell is actually run.
    family = args.map.build(args, values)
    # The typed starts, then each named set's, in the order the flags were given within each kind.
    # A sweep given neither is refused by Sweep, which already refuses a sweep with no starts.
    sweep = Sweep(family.spec, values, (*args.start, *named_starts(args.starts)), args.steps)
    home = sweep.home(SWEEPS)
    left = pending(sweep, home)
    print(f"sweep {home}")
    print(described(sweep, left), flush=True)
    # Before the manifest is written, because --status is a question: one that left a directory
    # behind would be an answer that had changed what it just measured, and every mistyped or
    # exploratory query would leave an empty sweep nothing can tell from an abandoned one.
    if args.status:
        return 0
    # Every start with work left, like every value, is offered to the map before anything is
    # written: a start it cannot step is a sweep that dies partway and dies in the same place on
    # every resume. Below the return above rather than beside the values, because a model family
    # answers this with the checkpoint, and `--status` runs no cell and should load nothing.
    #
    # [LAW:dataflow-not-control-flow] the starts of the cells about to run, not the ones the
    # command line named, so a finished sweep asks about none and a family with nothing to answer
    # loads nothing: rerunning a finished model sweep to see that it is finished would otherwise
    # read half a billion parameters to print a number it already has - and, off Metal, would
    # raise where it should have answered.
    family.holds(tuple(dict.fromkeys(cell.start for cell in left)))  # each start once, in cell order
    # And the map itself, once, at the same moment and for the same reason. What a map is made out
    # of - a steering direction's layer, the length of its vector - is the family's and not the
    # value's, so a family that cannot make one here cannot make one at any cell of this sweep.
    # [LAW:parse-dont-validate] past this line a family that can make maps exists, which is what
    # lets the line below write a directory: refused here the refusal costs nothing, and refused
    # one line later it has already left a sweep directory no run will ever fill - the litter the
    # `--status` return above exists to avoid. Guarded on `left` as `holds` is, so rerunning a
    # finished sweep to see that it is finished still loads nothing.
    if left:
        family.at(left[0].value)
    write_sweep(sweep, SWEEPS)
    failures: list[Failed] = []
    # [LAW:no-silent-failure] the count and the refusals are printed however the run ends, because
    # they are about what it did rather than about how it stopped. Without this a cell refused at
    # 3 and a map that misnames its file at 4 would end with only the second said out loud, and
    # the first would survive as one line in the scrollback of a sweep that prints thousands.
    try:
        for done, cell in enumerate(left, start=1):
            # [LAW:dataflow-not-control-flow] one line per cell whatever came of it, so the
            # progress a person watches scroll past has one shape: the value it ran at, the cell
            # it is, and what stopped it if anything did. The two arms are what running a cell can
            # come to, as `verdict`'s three are what the detector can see.
            outcome = run_cell(family, cell, sweep.steps)
            match outcome:
                case Trajectory() as trajectory:
                    write_trajectory(trajectory, home)
                    note = ""
                case Failed() as failure:
                    failures.append(failure)
                    note = f"  cannot run: {failure.reason}"
                case _:  # a third outcome would otherwise leave the line below printing a stale note
                    assert_never(outcome)
            # The cell's own name and not the trajectory's, though `run_cell` has just proved them
            # equal: it is what the cell is, so a refused cell is named here as exactly as a
            # written one. The value beside it is rounded to fit a column a thousand of these
            # scroll through, and rounded it names a different orbit - so it reads the line, and
            # the name identifies it. [LAW:one-source-of-truth]
            print(f"{done:>5}/{len(left)}  value {cell.value:<12.6g} {cell.name}{note}", flush=True)  # a --remote run streams through a pipe
    finally:
        # Said twice on purpose: inline, where a person watching sees which cell it was, and again
        # here, where it survives a thousand lines of scrollback. The value is written as the
        # shortest text that reads back as itself, the rule `logistic_state` holds a state to: it
        # is what names the cell, and rounded to six figures it names a different orbit in a
        # different file.
        #
        # Nothing here may raise, because a `finally` that raises replaces whatever ended the run:
        # with stdout closed by `uni sweep ... | head`, the SweepError above would reach the user as
        # a traceback instead of as `uni: ...` and EXIT_CONFIG, and no handler further out can
        # recover an exception this block has already destroyed. `say` already answers for a
        # stream that cannot carry its message; the count is suppressed on the same terms - OSError
        # and not BrokenPipeError, because a full disk under `> file` loses a report the way `| head`
        # does. It is only ever the report that is dropped - `pending` reads a directory through
        # `Path.exists`, which answers rather than raises.
        for failure in failures:
            say(f"value {failure.cell.value!r} from {failure.cell.start!r} has no orbit: {failure.reason}")
        with contextlib.suppress(OSError):
            print(described(sweep, pending(sweep, home)), flush=True)
    return EXIT_INCOMPLETE if failures else 0


FIGURES = Path("figures")  # committed, unlike trajectories and sweeps: a figure is what a person looks at
CURVES = Path("curves")  # committed too: a curve is what every map fitted to it is fitted to, and costs a forward pass a push to read again


class PlotError(ConfigError):
    """A sweep cannot be drawn. The message says what is not there to draw."""


def run_plot(args: argparse.Namespace) -> int:
    """Draw a persisted sweep's two pictures, and say where each one went."""
    from uni.draw import scatter
    from uni.figure import orbit_diagram, read, return_map
    from uni.sweep import MANIFEST, read_sweep

    # The manifest is read from the directory rather than described again on the command line: the
    # directory is named by the hash of those bytes, so the two cannot disagree about which sweep
    # this is. [LAW:one-source-of-truth]
    sweep = read_sweep(args.sweep / MANIFEST)
    series = read(sweep, args.sweep, args.observable, args.burn_in)
    total = len(sweep.values) * len(sweep.starts)
    pictures = {"return": return_map(series, args.observable), "orbit": orbit_diagram(series, args.observable)}
    # [LAW:no-silent-failure] an empty picture is a file that looks like an answer, so it is the
    # pictures that are checked and not the readings behind them: the return map needs two
    # readings out of one cell where the orbit diagram needs one, so a burn-in leaving exactly one
    # step draws a full orbit diagram beside a blank return map. Both are made before either is
    # written, so a sweep that can only answer for one of them leaves neither behind.
    bare = [kind for kind, picture in pictures.items() if not picture.points]
    if bare:
        raise PlotError(
            f"nothing to draw the {' and '.join(bare)} map of: {len(series)} of "
            f"{total} cells are on disk, and --burn-in {args.burn_in} "
            f"leaves {sum(len(one.numbers) for one in series)} readings across them - a return map "
            "needs two from one cell, an orbit diagram one"
        )
    for kind, picture in pictures.items():
        # Named by the sweep and what was read off it, so two sweeps and two observables are four
        # files rather than one overwritten four times. The burn-in is in it because it is the
        # third thing that fixes what the picture shows, and this repo names a file by what fixes
        # it: without it, plotting the reference cascade again without `--burn-in` would replace
        # the committed figure with a transient-laden one under the name the README links to.
        # The name comes from the manifest and not
        # from the directory it was found in: they agree for a sweep this program wrote, and when
        # they do not - a copied directory, a renamed one - it is the sweep that says which
        # picture this is. [LAW:one-source-of-truth]
        stem = f"{sweep.name}-{args.observable.replace(':', '-')}-burn{args.burn_in}-{kind}"
        drawn = scatter(picture, args.out / f"{stem}.png")
        # [LAW:no-silent-failure] the count on the refusal below says how much of the sweep is
        # there, and the success line said nothing - so a picture of twelve cells out of six
        # hundred looked exactly like a picture of all of them, under the same name.
        print(f"{drawn}  {len(picture.points)} points from {len(series)} of {total} cells", flush=True)
    return 0


def run_response(args: argparse.Namespace) -> int:
    """Print the model's own response to each push on the grid at each layer, and draw the curves."""
    import hashlib
    import json
    from dataclasses import asdict

    from uni.curve import write_curves
    from uni.draw import scatter
    from uni.figure import Picture, Point
    from uni.model import Model
    from uni.pinned import load_pinned
    from uni.response import admit, response, turns
    from uni.steer import Steer, read_direction
    from uni.sweep import grid

    values = grid(args.grid)
    prompt = template(args.template)
    pinned = load_pinned()
    steer = Steer(read_direction(args.knob, pinned))
    model = Model(pinned)
    # Every cell put to the refusals before the first forward pass, rather than met at its turn: a
    # layer the response cannot be read at, or a push too large to read, is wrong before anything
    # is computed, and the table is printed only once every curve is done.
    rounding = max(admit(model, steer, value, layer) for layer in args.layer for value in values)
    curves = {layer: tuple(response(model, prompt.render(args.start), steer, value, layer) for value in values) for layer in args.layer}
    name = steer.direction.contrast.name
    print(f"{pinned.dtype} rounding of the push moves no reading below by more than {rounding:.2g}")
    print(f"{'value':>10}" + "".join(f"{f'layer {layer}':>12}" for layer in curves))
    for i, value in enumerate(values):
        print(f"{value:>10.4g}" + "".join(f"{curve[i]:>12.4f}" for curve in curves.values()))
    for layer, curve in curves.items():
        top = max(range(len(values)), key=curve.__getitem__)
        print(f"layer {layer}: maximum {curve[top]:.4f} at {values[top]:.4g}; the slope changes sign at {list(turns(values, curve))}")
    described = {"pinned": asdict(pinned), "template": prompt.text, "start": args.start, "knob": steer.spec}
    print(write_curves(described, values, curves, steer.direction.squared_length, args.curves))
    # Named for everything that fixes the curves, as a sweep's pictures are named for the sweep.
    fixed = {**described, "values": list(values), "layers": list(curves)}
    stem = hashlib.sha256(json.dumps(fixed, sort_keys=True).encode()).hexdigest()[:16]
    points = tuple(Point(value, reading, layer) for layer, curve in curves.items() for value, reading in zip(values, curve))
    picture = Picture(points, f"response along {name}", f"push along {name}", f"what the layers after the push write along {name}", "layer" if len(curves) > 1 else None)
    print(scatter(picture, args.out / f"response-{stem}.png"))
    return 0


def run_smooth(args: argparse.Namespace) -> int:
    """Print how far a curve's readings lie from the series of each degree fitted to them, beside their own jitter."""
    from uni.curve import read_curve
    from uni.smooth import jitter, smooth

    curve = read_curve(args.curve, args.layer)
    print(f"curve {curve.name} at layer {curve.layer}: {len(curve.values)} readings from {curve.values[0]:g} to {curve.values[-1]:g}, jitter {jitter(curve.readings):.2e}")
    for degree in args.degree:
        fitted = smooth(curve, degree)
        print(f"degree {degree:>4}: rms residual {fitted.residual:.2e}, largest {fitted.largest:.2e}", flush=True)
    return 0


def bracket(text: str) -> tuple[float, float]:
    """LOW:HIGH, two finite numbers in order: where `uni fixed` looks for a fixed point."""
    parts = text.split(":")
    try:
        low, high = (finite(part) for part in parts) if len(parts) == 2 else (math.nan, math.nan)
    except (ValueError, argparse.ArgumentTypeError):
        low = high = math.nan
    if not low < high:  # nan fails this too
        raise argparse.ArgumentTypeError(f"a bracket is LOW:HIGH, two finite numbers with LOW below HIGH; got {text!r}")
    return low, high


def numbers_of(family: Family, command: str) -> Numbers:
    """How the family's states read as numbers and are written back, refused for a family whose states are texts.

    [LAW:single-enforcer] the one place a command that does arithmetic on states learns it can:
    a fixed point, a slope and a superstable value are all numbers, and a map whose states are
    texts has none of them, so it is told so rather than asked to spell a midpoint.
    """
    from uni.maps import NUMBERS, MapError

    kind = family.spec["kind"]
    if kind not in NUMBERS:
        raise MapError(f"uni {command} reads a map whose states are numbers, and the {kind} map's are not; the maps whose are: {', '.join(NUMBERS)}")
    return NUMBERS[kind](family.spec)


def run_fixed(args: argparse.Namespace) -> int:
    """Print the fixed point and the map's slope there at each value on the grid, and where the slope passes through -1."""
    from uni.fixed import crossings, fixed_point, slope
    from uni.sweep import grid

    values = grid(args.grid)
    family = args.map.build(args, values)
    numbers = numbers_of(family, "fixed")
    low, high = args.bracket
    print(f"{'value':>10}  {'fixed point':>14}  {'slope':>10}", flush=True)
    slopes = []
    for value in values:
        map = family.at(value)
        point = fixed_point(map, numbers, low, high)
        slopes.append(slope(map, numbers, point, args.step))
        print(f"{value:>10.6g}  {numbers.write(point):>14}  {slopes[-1]:>10.4f}", flush=True)
    # -1 is where a fixed point gives way to a period-2 orbit, and +1 where it is born or dies
    # beside another; both are said, since a map need not meet the first before the second.
    for level in (-1.0, 1.0):
        found = crossings(values, slopes, level)
        where = f"passes through {level:+g} at {', '.join(f'{value:.6g}' for value in found)}" if found else f"does not pass through {level:+g} on this grid"
        print(f"the slope {where}")
    return 0


def run_cascade(args: argparse.Namespace) -> int:
    """Print the superstable value on each grid, one period doubled per grid, and the ratios of their spacings."""
    from uni.cascade import nearest, quotients, ratios, returns, superstable
    from uni.fit import crossing
    from uni.sweep import grid

    grids = tuple(grid(text) for text in args.grid)
    family = args.map.build(args, tuple(value for values in grids for value in values))
    numbers = numbers_of(family, "cascade")
    family.holds((args.critical,))
    print(f"{'period':>6}  {'superstable value':>18}  {'error':>8}  {'scatter':>8}  {'nearest point':>15}  {'error':>8}", flush=True)
    periods = tuple(args.period * 2**doubling for doubling in range(len(grids)))
    found, distances = [], {}
    for period, values in zip(periods, grids):
        returned = [returns(family.at(value), numbers, args.critical, period) for value in values]
        parabola = superstable(values, returned, period)
        found.append(crossing(parabola, 0))
        row = f"{period:>6}  {found[-1].value:>18.10g}  {found[-1].error:>8.1e}  {parabola.scatter:>8.1e}"
        # An odd period has no point half a period round; the first of a cascade from an odd one is its only such row.
        if period % 2 == 0:
            distances[period] = nearest(values, returned, period, found[-1])
            row += f"  {distances[period].value:>15.8e}  {distances[period].error:>8.1e}"
        print(row, flush=True)
    for index, ratio in enumerate(ratios(found)):
        # Every digit a float gives, and the error beside it, as the values above are printed: which
        # of those digits mean anything is the error's to say, and a smooth map's say more than four do.
        print(f"spacing ratio over periods {', '.join(map(str, periods[index : index + 3]))}: {ratio.value:.7f} +- {ratio.error:.1e}")
    for (period, _), quotient in zip(distances.items(), quotients(tuple(distances.values()))):
        print(f"nearest-point ratio over periods {period}, {2 * period}: {quotient.value:.7f} +- {quotient.error:.1e}")
    return 0


def run_critical(args: argparse.Namespace) -> int:
    """Print where the map at one value is highest or lowest on a grid of states: the top a cascade is read from."""
    from uni.fit import crossing, fit
    from uni.sweep import grid

    value = args.map.value(args.value)
    family = args.map.build(args, (value,))
    numbers = numbers_of(family, "critical")
    states = tuple(numbers.write(point) for point in grid(args.grid))
    family.holds(states)
    map = family.at(value)
    # A cubic, the lowest degree that can lean: a top that falls away faster on one side than the
    # other pulls a parabola's vertex toward the gentler side, by more the wider the grid.
    cubic = fit(tuple(numbers.read(state) for state in states), tuple(numbers.read(map.step(state)) for state in states), 3)
    top = crossing(cubic, 1)
    print(f"the map at {value:g} turns at {top.value:.8g} +- {top.error:.1e} (a cubic through {len(states)} states, scatter {cubic.scatter:.1e}), written {numbers.write(top.value)}")
    return 0

def verdict(period: Period) -> str:
    """What the detector saw, in a sentence. The one branch is the domain's own three answers."""
    match period:
        case Cycle(length=length, onset=onset):
            return f"period {length}, entered at step {onset}"
        case NoCycle(examined=examined):
            # At least: showing a period of P takes P + 1 states, so a window of N that shows no
            # repeat leaves a period of exactly N standing. Claiming "longer" would rule it out.
            return f"no period: {examined} steps examined and no state repeated, so any period is at least {examined}"
        case Contradiction(onset=onset, length=length, step=step):
            return (
                f"the state at step {onset} came back {length} steps later, but step {step} is not the state "
                f"{length} steps before it. The map is not a function of its state; run `uni determinism` "
                "before reading anything into this orbit"
            )
        case _:  # a fourth answer would otherwise be printed as the word None
            assert_never(period)


def run_observe(args: argparse.Namespace) -> int:
    """Read a written trajectory back: each step's observables, and the period of its orbit."""
    from uni.loop import read_trajectory
    from uni.observe import Weights, identities, observables, readings, steps
    from uni.period import detect

    trajectory = read_trajectory(args.trajectory)
    # The start is step 0 of the orbit, as `uni loop` prints it, so it is numbered and
    # searched with the rest: an orbit that comes back to the text it started from has a
    # period through step 0, and leaving the start out would hide exactly that.
    orbit = (trajectory.start, *trajectory.states)
    numbers = identities(orbit)
    # The period is read off the states alone, so it is printed before the checkpoint is even
    # loaded: it is the answer, the rows below are the evidence, and no step that cannot be
    # scored can take it away.
    print(verdict(detect(orbit, args.burn_in)), flush=True)  # a --remote run's stdout is a pipe, not a terminal
    print()
    columns = observables(trajectory, Weights())
    print(f"{'step':>4}  {'state':>5}" + "".join(f"  {column.name:>16}" for column in columns), flush=True)
    # The start was given rather than stepped into, so no observable of a step has a reading for
    # it; its row is printed anyway, so the identity column reads as the orbit and every step the
    # verdict can name is one the table shows.
    print(f"{0:>4}  {numbers[0]:>5}" + "".join(f"  {'-':>16}" for _ in columns))
    for step, number in zip(steps(trajectory), numbers[1:]):
        row = "".join(f"  {reading:>16.6f}" for reading in readings(columns, step))
        print(f"{step.index:>4}  {number:>5}{row}", flush=True)  # a --remote run streams through a pipe
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uni",
        description="Feigenbaum universality in LLM feedback loops.",
        parents=[REMOTE],
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    host = commands.add_parser("host", help="print where uni is running")
    host.set_defaults(run=run_host)
    gen = commands.add_parser("gen", help="generate greedily from the pinned model")
    gen.add_argument("prompt")
    gen.set_defaults(run=run_gen)
    determinism = commands.add_parser("determinism", help="generate each gate case many times and check every hash is equal")
    determinism.add_argument("--runs", type=positive, default=RUNS, help=f"runs per case (default: {RUNS})")
    determinism.set_defaults(run=run_determinism)
    loop = commands.add_parser("loop", parents=[MAP_FLAGS], help="iterate a map from a start state and write the trajectory")
    loop.add_argument("--map", type=map_named, default="model", help=f"the map to iterate: {', '.join(MAPS)} (default: model)")
    loop.add_argument("--start", required=True, help="the first state; may be empty; write --start=TEXT when it begins with '-'")
    loop.add_argument("--steps", type=positive, required=True, help="how many times to step the map")
    loop.add_argument("--value", type=finite, help="the map's parameter: the knob's setting (default: 0), r, or the response map's gain; the last two have no default")
    loop.set_defaults(run=run_loop)
    sweep = commands.add_parser("sweep", parents=[MAP_FLAGS], help="run a map at every value on a grid, from every start, and keep the orbits")
    sweep.add_argument("--map", type=map_named, default="model", help=f"the map to sweep: {', '.join(MAPS)} (default: model)")
    sweep.add_argument("--grid", required=True, help="the values to sweep, as FROM:TO:COUNT with TO included, as in 2.8:4.0:200")
    sweep.add_argument("--start", action="append", default=[], help="a start state; repeat it for each one, and write --start=TEXT when it begins with '-'")
    sweep.add_argument("--starts", action="append", default=[], help="a set of starts named in uni/starts.toml, run after any --start; repeat it for each set")
    sweep.add_argument("--steps", type=positive, required=True, help="how many times to step the map in each cell")
    sweep.add_argument("--status", action="store_true", help="say how many cells are done and run none of them")
    sweep.set_defaults(run=run_sweep)
    observe = commands.add_parser("observe", help="read a written trajectory's observables and the period of its orbit")
    observe.add_argument("trajectory", type=Path, help="a trajectory file written by `uni loop`")
    observe.add_argument("--burn-in", type=whole, default=0, dest="burn_in", help="steps to pass over before looking for a period (default: 0)")
    plot = commands.add_parser("plot", help="draw a persisted sweep's return map and orbit diagram")
    plot.add_argument("sweep", type=Path, help="a sweep directory written by `uni sweep`")
    plot.add_argument("--observable", required=True, help="the number to read off each step, as `uni observe` names its columns")
    plot.add_argument("--burn-in", type=whole, default=0, dest="burn_in", help="steps to pass over before the orbit is taken as settled (default: 0)")
    plot.add_argument("--out", type=Path, default=FIGURES, help=f"where to write the figures (default: {FIGURES})")
    plot.set_defaults(run=run_plot)
    observe.set_defaults(run=run_observe)
    fixed = commands.add_parser("fixed", parents=[MAP_FLAGS], help="find where a map of numbers holds still at each value, and its slope there")
    fixed.add_argument("--map", type=map_named, required=True, help=f"the map to read: {', '.join(MAPS)}; its states must be numbers")
    fixed.add_argument("--grid", required=True, help="the values, as FROM:TO:COUNT with TO included")
    fixed.add_argument("--bracket", type=bracket, required=True, help="LOW:HIGH, states the map carries in opposite directions; write --bracket=LOW:HIGH when LOW is negative")
    fixed.add_argument("--step", type=finite, required=True, help="the half-width of the central difference the slope is read across, in the state's own units")
    fixed.set_defaults(run=run_fixed)
    cascade = commands.add_parser("cascade", parents=[MAP_FLAGS], help="find the value on each grid where the orbit from the map's top returns to itself, and the ratios of their spacings")
    cascade.add_argument("--map", type=map_named, required=True, help=f"the map to read: {', '.join(MAPS)}; its states must be numbers")
    cascade.add_argument("--critical", required=True, help="the state at the map's top, spelled as the map writes it; write --critical=STATE when it begins with '-'")
    cascade.add_argument("--period", type=positive, required=True, help="the period whose superstable value the first grid holds; each later grid holds twice the one before")
    cascade.add_argument("--grid", action="append", required=True, help="values around one superstable value, as FROM:TO:COUNT with TO included; repeat it for each period, in order")
    cascade.set_defaults(run=run_cascade)
    critical = commands.add_parser("critical", parents=[MAP_FLAGS], help="find where a map of numbers is highest or lowest on a grid of states, at one value")
    critical.add_argument("--map", type=map_named, required=True, help=f"the map to read: {', '.join(MAPS)}; its states must be numbers")
    critical.add_argument("--value", type=finite, help="the map's parameter to read it at, as `uni loop` takes it")
    critical.add_argument("--grid", required=True, help="the states around the top, as FROM:TO:COUNT with TO included; write --grid=FROM:TO:COUNT when FROM is negative")
    critical.set_defaults(run=run_critical)
    answer = commands.add_parser("response", help="read how the layers after a steering push answer it, with no token generated")
    answer.add_argument("--template", required=True, help="the template the start is rendered into, named in uni/templates.toml")
    answer.add_argument("--knob", required=True, help="the direction in uni/directions to push along")
    answer.add_argument("--start", required=True, help="the text the prompt is made from; write --start=TEXT when it begins with '-'")
    answer.add_argument("--grid", required=True, help="the pushes, as FROM:TO:COUNT with TO included")
    answer.add_argument("--layer", type=whole, action="append", required=True, help="a layer to read the response at; repeat it for each one")
    answer.add_argument("--out", type=Path, default=FIGURES, help=f"where to write the figure (default: {FIGURES})")
    answer.add_argument("--curves", type=Path, default=CURVES, help=f"where to write the curves, for `uni smooth` and the smooth map to read (default: {CURVES})")
    answer.set_defaults(run=run_response)
    smoothed = commands.add_parser("smooth", help="fit a series of each degree to a curve `uni response` wrote, and print how closely each fits beside the curve's own jitter")
    smoothed.add_argument("--curve", type=Path, required=True, help="a curve file `uni response` wrote")
    smoothed.add_argument("--layer", type=whole, required=True, help="the layer in it to fit")
    smoothed.add_argument("--degree", type=positive, action="append", required=True, help="a degree to fit; repeat it for each one")
    smoothed.set_defaults(run=run_smooth)
    direction = commands.add_parser("direction", help="derive a steering direction from uni/directions/<name>.toml")
    direction.add_argument("contrast", type=contrast, help="the contrast's name")
    direction.set_defaults(run=run_direction)
    return parser


def checkout_root(cwd: Path) -> Path:
    # [LAW:parse-dont-validate] the tree to sync is the checkout you are standing in, as git
    # sees it, never the location of an installed copy of this package.
    found = subprocess.run(["git", "-C", str(cwd), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if found.returncode:
        raise RemoteConfigError(f"--remote runs from inside a checkout of the repo; {cwd} is not one")
    return Path(found.stdout.strip())


# The commands whose answer is a file in this checkout, so there is nowhere on the host to put it.
# Named here rather than read off the parsed command, which is what this first tried: parsing runs
# the converters, one of which reads a direction and so imports torch, and `--remote` exists
# precisely so this machine never pays for that. A set of names costs nothing to consult, and what
# keeps it in step with the parser below is a test that enumerates the parser's own subcommands -
# a name in here that no command answers to is a guard that silently stops guarding. Each name
# carries what to do instead, which differs by what the command reads.
HERE = {
    "plot": "bring the sweep home first (see the README) and run it without --remote",
    "response": "run it without --remote; the table it prints travels back, the figure and the curves it writes would not",
}


def stays_here(rest: Sequence[str]) -> bool:
    """Whether the command in `rest` is one whose answer is a file in this checkout.

    Read off the front of what is left after `--remote` is taken out, which is where the
    subcommand is: the top-level parser carries no other argument that takes a value.
    """
    return bool(rest) and rest[0] in HERE


def run(argv: Sequence[str], env: Mapping[str, str], cwd: Path) -> int:
    """Do what the command line asks. How it says that it could not is `main`'s, not this one's.

    Split from `main` so that turning a failure into an exit code happens in one place, and so
    that place covers the remote leg as well as the local one. [LAW:effects-at-boundaries]
    """
    remote, rest = split_remote(argv)
    if remote and stays_here(rest):
        # [LAW:no-silent-failure] the sync has one leg and figures/ is not on it, so this would
        # draw on the host, print a path that does not exist here, and exit 0 as though it had
        # answered. The sweep is the thing that travels; the picture is drawn where it is kept.
        raise ConfigError(f"{rest[0]} writes a file into this checkout, so it runs here, not on the host; {HERE[rest[0]]}")
    if not remote:
        args = build_parser().parse_args(rest)
        return args.run(args)
    tree = checkout_root(cwd)
    # The checkout's .env, under the real environment: a set variable wins over the file.
    dotenv = {k: v for k, v in dotenv_values(tree / ".env").items() if v is not None}
    return run_remote(remote_target_from_env({**dotenv, **env}), rest, tree)


def say(message: str) -> None:
    """Tell whoever is reading stderr why the run stopped, if anyone still is.

    The exit code is the answer and this line is its explanation, so a stderr that cannot carry it
    - `2>&1 | head` once head has gone - costs the explanation and not the answer. Raising here
    would cost both: a traceback with nowhere to go, and an exit code other than the one this
    run earned. [LAW:no-silent-failure] [LAW:single-enforcer] the one place a refusal is said.
    """
    with contextlib.suppress(OSError):
        print(f"uni: {message}", file=sys.stderr)


def main(argv: Sequence[str], env: Mapping[str, str], cwd: Path) -> int:
    """Run the command, and turn whatever stopped it into something the shell can read.

    [LAW:single-enforcer] the one place a failure becomes an exit code, so these three are the
    three answers this program has. They are three rather than one because what a reader does
    about them differs: change what you asked for, fix the machine, or nothing at all. Anything
    else reaching here is a bug, and raises on to `entry`, which gives it a code of its own.
    """
    try:
        code = run(argv, env, cwd)
        # The last of a successful run's output is written here rather than left to interpreter
        # exit, which answers a failure to deliver it with exit 120. For a success the output is the
        # answer, so output that never arrived is a full disk's EXIT_IO or a closed pipe's
        # EXIT_PIPE. A run that already has another answer keeps it: a diverged gate or a sweep
        # come up short outranks a report nobody received, as what ended a run outranks how far it
        # got, and `entry` sees to what could not be delivered.
        if code == 0:
            sys.stdout.flush()
        return code
    except ConfigError as error:
        say(str(error))
        return EXIT_CONFIG
    except BrokenPipeError:
        # Before OSError below, which it is one of, and answered rather than reported: a consumer
        # that stops reading is ordinary shell usage and not a fault of this run.
        return EXIT_PIPE
    except OSError as error:
        # The environment, reported in the OS's own words: `str` on an OSError already names the
        # errno, what it means, and the file it was about, which is the whole of what a person
        # fixes. Reported and not raised onward, because a machine that will not do the work is
        # not a bug in the program that asked. [LAW:no-silent-failure]
        say(str(error))
        return EXIT_IO


def entry() -> None:
    # A stream closed before this process started (`uni host >&-`) is None to Python. print
    # tolerates that and nothing else here does, so it becomes what the caller asked for - a
    # stream to nowhere - once, here, and every write past this line has a stream to write to.
    # [LAW:parse-dont-validate]
    sys.stdout = sys.stdout or open(os.devnull, "w")
    sys.stderr = sys.stderr or open(os.devnull, "w")
    try:
        sys.exit(main(sys.argv[1:], os.environ, Path.cwd()))
    except Exception:
        # Here and not in `main`, which a test calls in-process and wants a bug raised out of. The
        # traceback is the report, so it is printed as the interpreter would have; only the code
        # changes. Suppressed as `say` is, because a stderr that cannot carry it costs the
        # traceback and must not cost the code as well.
        with contextlib.suppress(OSError):
            traceback.print_exc()
        sys.exit(EXIT_BUG)
    finally:
        # Python flushes both streams again on its way out, and one that cannot take what is left
        # in it prints "Exception ignored in: <_io.TextIOWrapper ...>" over whatever the user piped
        # into and exits 120 in place of the code this run chose. What is left belongs to a run
        # already answered for - by `main`, which flushed its own output, or by argparse, whose
        # --help and usage errors leave through SystemExit and which drops a failed write itself -
        # so it has nowhere to go. A `finally` because both of those ways out need it, and safe as
        # one because nothing in it can raise over the exit already in flight. Here and not in
        # `main` because it rewires this process's own descriptors, which a caller of `main` in the
        # same process still needs. [LAW:effects-at-boundaries]
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except OSError:
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, stream.fileno())
                os.close(devnull)
