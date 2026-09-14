"""The `uni` command line. Every subcommand accepts --remote."""

from __future__ import annotations

import argparse
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from dotenv import dotenv_values

from uni.remote import RemoteConfigError, remote_target_from_env, run_remote

# The repo root when uni is installed editable from a checkout: what --remote syncs, and where .env lives.
ROOT = Path(__file__).resolve().parents[1]

EXIT_CONFIG = os.EX_CONFIG  # distinct from argparse's 2 and from anything rsync or ssh returns


def checkout_root() -> Path:
    # [LAW:parse-dont-validate] a wheel install has no tree to sync; say so instead of mirroring site-packages.
    if not (ROOT / "pyproject.toml").is_file():
        raise RemoteConfigError(f"--remote runs from a checkout of the repo; this uni is installed at {ROOT}")
    return ROOT


def run_host(args: argparse.Namespace) -> int:
    """Print where this process is running, so a --remote run is visibly remote."""
    print(f"{platform.node()} {platform.machine()} {platform.system()} python {platform.python_version()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    remote = argparse.ArgumentParser(add_help=False)
    remote.add_argument(
        "--remote",
        action="store_true",
        help="sync this working tree to the run host and run the command there",
    )
    # allow_abbrev off: the flag is forwarded by exact match, so an abbreviation must not parse as it.
    parser = argparse.ArgumentParser(
        prog="uni", description="Feigenbaum universality in LLM feedback loops.", allow_abbrev=False
    )
    commands = parser.add_subparsers(dest="command", required=True)
    host = commands.add_parser("host", parents=[remote], allow_abbrev=False, help="print where uni is running")
    host.set_defaults(run=run_host)
    return parser


def main(argv: Sequence[str], env: Mapping[str, str]) -> int:
    args = build_parser().parse_args(argv)
    if not args.remote:
        return args.run(args)
    try:
        target = remote_target_from_env(env)
        tree = checkout_root()
    except RemoteConfigError as error:
        print(f"uni: {error}", file=sys.stderr)
        return EXIT_CONFIG
    # Forward what was typed, minus the flag that brought us here, so the host runs it locally.
    return run_remote(target, [arg for arg in argv if arg != "--remote"], tree)


def environment() -> dict[str, str]:
    """The process environment over the repo's .env; a real variable wins over the file."""
    dotenv = {k: v for k, v in dotenv_values(ROOT / ".env").items() if v is not None}
    return {**dotenv, **os.environ}


def entry() -> None:
    sys.exit(main(sys.argv[1:], environment()))
