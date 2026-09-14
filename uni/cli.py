"""The `uni` command line. Every subcommand accepts --remote."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from dotenv import dotenv_values

from uni.remote import RemoteConfigError, remote_target_from_env, run_remote

EXIT_CONFIG = os.EX_CONFIG  # distinct from argparse's 2 and from anything rsync or ssh returns

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
