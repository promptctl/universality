"""The `uni` command line. Every subcommand accepts --remote."""

from __future__ import annotations

import argparse
import math
import os
import platform
import subprocess
import sys
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
    from uni.maps import Family, Knob
    from uni.steer import Contrast

EXIT_CONFIG = os.EX_CONFIG  # distinct from argparse's 2 and from anything rsync or ssh returns
EXIT_DIVERGED = 1  # the determinism gate ran and some case produced more than one hash

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


# The flags that describe the model's map and no other one. Declared here rather than on `uni
# loop` directly so the set has a single spelling: every other map's builder refuses whatever
# this parser holds, so a flag added to it is refused by them without anything else being
# edited. A model-only flag listed nowhere is one a logistic run would accept and quietly
# ignore, which is the failure the refusal exists to prevent. [LAW:one-source-of-truth]
MODEL_FLAGS = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
MODEL_FLAGS.add_argument("--template", help="the model map's template, named in uni/templates.toml")
MODEL_FLAGS.add_argument("--knob", help="a direction in uni/directions for the model map to steer along, or none")
MODEL_ONLY = tuple(vars(MODEL_FLAGS.parse_args([])))  # the flags above, under the names args carries them by


def model_map(args: argparse.Namespace, values: Sequence[float]) -> Family:
    """The pinned model reading its prompt from a template, turned by the knob.

    [LAW:single-enforcer] every flag describing this map is read here, and only here. argparse
    cannot know which flags belong to the --map it was handed, so a converter on one of them
    would read a direction file, and load torch to hold it, for a run about to be refused.
    """
    from uni.maps import MapError, ModelFamily, NoKnob  # torch-free, so a refusal below costs nothing

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
    from uni.maps import LogisticFamily, MapError

    # [LAW:no-silent-failure] these describe the model's map. Carried over from an earlier command
    # and dropped without a word, they would read back as settings this run had honoured.
    for flag in MODEL_ONLY:
        if getattr(args, flag) is not None:
            raise MapError(f"--{flag} describes the model map; the logistic map's one parameter is the value, which is r")
    family = LogisticFamily()
    # [LAW:no-silent-failure] every value on the grid is offered to the map up front, as every
    # value is offered to the knob above. A sweep to r = 4.2 would otherwise write its manifest,
    # grind through a hundred and seventy cells and die on the first r the map refuses - and every
    # resume would march to the same wall again, against a total it can never reach.
    for value in values:
        family.at(value)
    return family


def model_value(given: float | None) -> float:
    """Unturned is a setting like any other, and the one a run that named no value means: at 0 a
    knob adds nothing, so the orbit is exactly the unsteered one."""
    return 0.0 if given is None else given


def logistic_value(given: float | None) -> float:
    """r has no default worth having. 0 is a legal r, so a forgotten --value would not fail: it
    would run the map that sends every state to zero and be answered `period 1` - a period claim,
    which is this project's whole output, about a parameter nobody chose."""
    from uni.maps import MapError

    if given is None:
        raise MapError("the logistic map's parameter is r; pass --value")
    return given


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


MAPS = {"model": Kind(model_map, model_value), "logistic": Kind(logistic_map, logistic_value)}


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
    from uni.loop import Trajectory, orbit, write_trajectory
    from uni.sweep import Sweep, SweepError, described, grid, pending, write_sweep

    # Read here and not by argparse: a grid that is not one is the user's typo, and this repo
    # reports that as `uni: ...` with EX_CONFIG. argparse catches only its own error type, so a
    # converter raising SweepError would reach the user as a traceback, which means a bug here.
    values = grid(args.grid)
    # Built once for the whole grid, which is the point of a family: the checkpoint behind a model
    # sweep is read once, and not until a cell is actually run.
    family = args.map.build(args, values)
    sweep = Sweep(family.spec, values, tuple(args.start), args.steps)
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
    write_sweep(sweep, SWEEPS)
    for done, cell in enumerate(left, start=1):
        map = family.at(cell.value)
        # The map's own description and its own value, the way `uni loop` records them, so what
        # the file says the orbit ran at is what it ran at. [LAW:one-source-of-truth]
        states = tuple(islice(orbit(map, cell.start), sweep.steps))
        trajectory = Trajectory(map.spec, map.value, cell.start, states)
        # [LAW:no-silent-failure] the sweep named this cell's file before running it, and
        # resumption is that name coming true. A map that described itself differently, or that
        # came back set to something other than what it was asked for, writes a file this sweep
        # cannot find - and then every rerun runs the cell again, for ever, against a total that
        # never lands.
        if trajectory.name != cell.name:
            raise SweepError(
                f"the cell at {cell.value} writes {trajectory.name}, not the {cell.name} this sweep is looking for: "
                "the map does not describe itself the same way twice"
            )
        write_trajectory(trajectory, home)
        print(f"{done:>5}/{len(left)}  value {cell.value:<12.6g} {cell.name}", flush=True)  # a --remote run streams through a pipe
    print(described(sweep, pending(sweep, home)))
    return 0


FIGURES = Path("figures")  # committed, unlike trajectories and sweeps: a figure is what a person looks at


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
    # [LAW:no-silent-failure] an empty picture is a file that looks like an answer. A sweep that
    # has run nothing yet, and a manifest sitting in some other sweep's directory, both land here.
    if not any(one.numbers for one in series):
        raise PlotError(
            f"{args.sweep} holds no readings to draw: {len(series)} of "
            f"{len(sweep.values) * len(sweep.starts)} cells are on disk, and --burn-in "
            f"{args.burn_in} leaves nothing of them"
        )
    for kind, picture in (("return", return_map(series, args.observable)), ("orbit", orbit_diagram(series, args.observable))):
        # Named for the sweep and what was read off it, so two sweeps and two observables are four
        # files rather than one overwritten four times.
        stem = f"{args.sweep.name}-{args.observable.replace(':', '-')}-{kind}"
        print(f"{scatter(picture, args.out / f'{stem}.png')}  {len(picture.points)} points", flush=True)
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
    loop = commands.add_parser("loop", parents=[MODEL_FLAGS], help="iterate a map from a start state and write the trajectory")
    loop.add_argument("--map", type=map_named, default="model", help=f"the map to iterate: {', '.join(MAPS)} (default: model)")
    loop.add_argument("--start", required=True, help="the first state; may be empty; write --start=TEXT when it begins with '-'")
    loop.add_argument("--steps", type=positive, required=True, help="how many times to step the map")
    loop.add_argument("--value", type=finite, help="the map's parameter: the knob's setting (default: 0), or r, which has no default")
    loop.set_defaults(run=run_loop)
    sweep = commands.add_parser("sweep", parents=[MODEL_FLAGS], help="run a map at every value on a grid, from every start, and keep the orbits")
    sweep.add_argument("--map", type=map_named, default="model", help=f"the map to sweep: {', '.join(MAPS)} (default: model)")
    sweep.add_argument("--grid", required=True, help="the values to sweep, as FROM:TO:COUNT with TO included, as in 2.8:4.0:200")
    sweep.add_argument("--start", action="append", required=True, help="a start state; repeat it for each one, and write --start=TEXT when it begins with '-'")
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


def main(argv: Sequence[str], env: Mapping[str, str], cwd: Path) -> int:
    remote, rest = split_remote(argv)
    if not remote:
        args = build_parser().parse_args(rest)
        try:
            return args.run(args)
        except ConfigError as error:  # [LAW:single-enforcer] the one type the CLI reports; a bug here is a traceback
            print(f"uni: {error}", file=sys.stderr)
            return EXIT_CONFIG
    try:
        tree = checkout_root(cwd)
        # The checkout's .env, under the real environment: a set variable wins over the file.
        dotenv = {k: v for k, v in dotenv_values(tree / ".env").items() if v is not None}
        target = remote_target_from_env({**dotenv, **env})
    except RemoteConfigError as error:
        print(f"uni: {error}", file=sys.stderr)
        return EXIT_CONFIG
    return run_remote(target, rest, tree)


def entry() -> None:
    sys.exit(main(sys.argv[1:], os.environ, Path.cwd()))
