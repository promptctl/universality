"""The `uni` command line. Every subcommand accepts --remote."""

from __future__ import annotations

import argparse
import math
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, assert_never

from dotenv import dotenv_values

from uni.determinism import RUNS
from uni.parse import ConfigError
from uni.period import Contradiction, Cycle, NoCycle, Period
from uni.remote import RemoteConfigError, remote_target_from_env, run_remote
from uni.template import Template, load_templates

if TYPE_CHECKING:
    from uni.maps import Knob
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


def knob(name: str) -> Knob:
    # Imported here: a steering knob holds torch tensors, and only `uni loop` pays for loading torch.
    from uni.maps import NoKnob
    from uni.pinned import load_pinned
    from uni.steer import Steer, read_direction

    if name == "none":
        return NoKnob()
    try:
        return Steer(read_direction(name, load_pinned()))
    except ConfigError as error:  # argparse prints a traceback for anything but its own error type
        raise argparse.ArgumentTypeError(str(error)) from error


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
    try:
        templates = load_templates()
    except ConfigError as error:  # argparse prints a traceback for anything but its own error type
        raise argparse.ArgumentTypeError(str(error)) from error
    if name not in templates:
        raise argparse.ArgumentTypeError(f"no template {name!r}; the templates are {', '.join(templates)}")
    return templates[name]


def run_loop(args: argparse.Namespace) -> int:
    """Print the start and every state as it lands, then write the trajectory file."""
    from uni.loop import Trajectory, orbit, write_trajectory
    from uni.maps import ModelMap
    from uni.model import Model
    from uni.pinned import load_pinned

    # [LAW:parse-dont-validate] the knob takes its value here, before a checkpoint is downloaded:
    # past this line a map exists, and a map exists only at a value its knob accepted.
    knob = args.knob.turn(args.value)
    map = ModelMap(Model(load_pinned()), args.template, knob)
    print(f"{'step':>4}  state")
    print(f"{0:>4}  {args.start!r}")
    states = []
    for step, state in enumerate(islice(orbit(map, args.start), args.steps), start=1):
        print(f"{step:>4}  {state!r}", flush=True)  # repr, so each state is one line and an empty one shows
        states.append(state)
    path = write_trajectory(Trajectory(map.spec, args.value, args.start, tuple(states)), TRAJECTORIES)
    print()
    print(f"trajectory {path}")
    return 0


def verdict(period: Period) -> str:
    """What the detector saw, in a sentence. The one branch is the domain's own three answers."""
    match period:
        case Cycle(length=length, onset=onset):
            return f"period {length}, entered at step {onset}"
        case NoCycle(examined=examined):
            return f"no period: {examined} steps examined and no state repeated, so any period is longer than that"
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
    from uni.model import Model
    from uni.observe import Length, Logprob, Projection, identities, readings, steering_additions, steering_directions, steps, template_of, written_by
    from uni.period import detect
    from uni.pinned import load_pinned

    trajectory = read_trajectory(args.trajectory)
    template = template_of(trajectory)
    pinned = written_by(trajectory, load_pinned())
    directions = steering_directions(trajectory, pinned)
    # The start is step 0 of the orbit, as `uni loop` prints it, so it is numbered and
    # searched with the rest: an orbit that comes back to the text it started from has a
    # period through step 0, and leaving the start out would hide exactly that.
    orbit = (trajectory.start, *trajectory.states)
    numbers = identities(orbit)
    # The period is read off the states alone, so it is printed before the checkpoint is even
    # loaded: it is the answer, the rows below are the evidence, and no step that cannot be
    # scored can take it away.
    print(verdict(detect(orbit, args.burn_in)))
    print()
    model = Model(pinned)
    observables = (
        Length(),
        Logprob(model, template, steering_additions(directions, trajectory.value)),
        *(Projection(model, template, direction) for direction in directions),
    )
    print(f"{'step':>4}  {'state':>5}" + "".join(f"  {observable.name:>16}" for observable in observables))
    # The start was given rather than stepped into, so no observable of a step has a reading for
    # it; its row is printed anyway, so the identity column reads as the orbit and every step the
    # verdict can name is one the table shows.
    print(f"{0:>4}  {numbers[0]:>5}" + "".join(f"  {'-':>16}" for _ in observables))
    for step, number in zip(steps(trajectory), numbers[1:]):
        row = "".join(f"  {reading:>16.6f}" for reading in readings(observables, step))
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
    loop = commands.add_parser("loop", help="feed the model its own output under a template and write the trajectory")
    loop.add_argument("--template", type=template, required=True, help="a template named in uni/templates.toml")
    loop.add_argument("--start", required=True, help="the first state; may be empty; write --start=TEXT when it begins with '-'")
    loop.add_argument("--steps", type=positive, required=True, help="how many times to step the map")
    loop.add_argument("--knob", type=knob, default="none", help="a direction in uni/directions to steer along, or none (default)")
    loop.add_argument("--value", type=finite, default=0.0, help="the knob's value (default: 0)")
    loop.set_defaults(run=run_loop)
    observe = commands.add_parser("observe", help="read a written trajectory's observables and the period of its orbit")
    observe.add_argument("trajectory", type=Path, help="a trajectory file written by `uni loop`")
    observe.add_argument("--burn-in", type=whole, default=0, dest="burn_in", help="steps to pass over before looking for a period (default: 0)")
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
