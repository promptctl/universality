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
from typing import TYPE_CHECKING

from dotenv import dotenv_values

from uni.determinism import RUNS
from uni.remote import RemoteConfigError, remote_target_from_env, run_remote
from uni.template import Template, TemplateError, load_templates

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
    from uni.steer import Steer, SteerError, read_direction

    if name == "none":
        return NoKnob()
    try:
        return Steer(read_direction(name, load_pinned()))
    except SteerError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def contrast(name: str) -> Contrast:
    from uni.steer import SteerError, load_contrast

    try:
        return load_contrast(name)
    except SteerError as error:
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
    except TemplateError as error:  # argparse would print a traceback, or hide the message behind its own
        raise argparse.ArgumentTypeError(str(error)) from error
    if name not in templates:
        raise argparse.ArgumentTypeError(f"no template {name!r}; the templates are {', '.join(templates)}")
    return templates[name]


def run_loop(args: argparse.Namespace) -> int:
    """Print the start and every state as it lands, then write the trajectory file."""
    if args.knob.spec is None and args.value:
        # [LAW:no-silent-failure] a recorded value nothing applied reads back as a steering run that did nothing.
        print(f"uni: --value {args.value} has nothing to turn; pass --knob, or leave --value at 0", file=sys.stderr)
        return EXIT_CONFIG
    from uni.loop import Trajectory, orbit, write_trajectory
    from uni.maps import ModelMap
    from uni.model import Model
    from uni.pinned import load_pinned

    map = ModelMap(Model(load_pinned()), args.template, args.knob)
    print(f"{'step':>4}  state")
    print(f"{0:>4}  {args.start!r}")
    states = []
    for step, state in enumerate(islice(orbit(map, args.value, args.start), args.steps), start=1):
        print(f"{step:>4}  {state!r}", flush=True)  # repr, so each state is one line and an empty one shows
        states.append(state)
    path = write_trajectory(Trajectory(map.spec, args.value, args.start, tuple(states)), TRAJECTORIES)
    print()
    print(f"trajectory {path}")
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
        return args.run(args)
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
