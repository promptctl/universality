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

# The repo root: what --remote syncs, and where .env lives.
ROOT = Path(__file__).resolve().parents[1]

EXIT_CONFIG = 2  # argparse's usage-error code, reused for a misconfigured host


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
    parser = argparse.ArgumentParser(prog="uni", description="Feigenbaum universality in LLM feedback loops.")
    commands = parser.add_subparsers(dest="command", required=True)
    host = commands.add_parser("host", parents=[remote], help="print where uni is running")
    host.set_defaults(run=run_host)
    return parser


def main(argv: Sequence[str], env: Mapping[str, str]) -> int:
    args = build_parser().parse_args(argv)
    if not args.remote:
        return args.run(args)
    try:
        target = remote_target_from_env(env)
    except RemoteConfigError as error:
        print(f"uni: {error}", file=sys.stderr)
        return EXIT_CONFIG
    # Forward what was typed, minus the flag that brought us here, so the host runs it locally.
    return run_remote(target, [arg for arg in argv if arg != "--remote"], ROOT)


def environment() -> dict[str, str]:
    """The process environment over the repo's .env; a real variable wins over the file."""
    dotenv = {k: v for k, v in dotenv_values(ROOT / ".env").items() if v is not None}
    return {**dotenv, **os.environ}


def entry() -> None:
    sys.exit(main(sys.argv[1:], environment()))
