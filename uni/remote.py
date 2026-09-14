"""Run a `uni` command on the experiment host.

The host's identity never lives in this tree. It arrives through the environment,
normally a gitignored `.env` (see `.env.example`), and is parsed once here into a
`RemoteTarget` that the rest of the program takes as a plain value.
"""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

VARIABLES = ("UNI_REMOTE_HOST", "UNI_REMOTE_USER", "UNI_REMOTE_DIR")

# What the sync carries is what git would: .gitignore decides, in one place.
# .git is not needed to run, and .env is pinned here because it holds the host's identity.
SYNC_FILTERS = ("--exclude=.git", "--exclude=.env", "--filter=:- .gitignore")


class RemoteConfigError(Exception):
    """The environment does not describe a usable host. The message says what to fix."""


@dataclass(frozen=True)
class RemoteTarget:
    host: str
    user: str
    dir: str  # absolute path on the host

    @property
    def ssh_target(self) -> str:
        return f"{self.user}@{self.host}"


def remote_target_from_env(env: Mapping[str, str]) -> RemoteTarget:
    # [LAW:parse-dont-validate] the one checkpoint: past here the host is a value, not
    # three strings that may or may not be set.
    missing = [name for name in VARIABLES if not env.get(name)]
    if missing:
        raise RemoteConfigError(
            f"{', '.join(missing)} missing or empty; copy .env.example to .env and fill it in"
        )
    host, user, dir = (env[name] for name in VARIABLES)
    if not dir.startswith("/"):
        raise RemoteConfigError(f"UNI_REMOTE_DIR must be an absolute path on the host, got {dir!r}")
    return RemoteTarget(host=host, user=user, dir=dir)


# [LAW:effects-at-boundaries] the two steps are pure descriptions; run_remote performs them.


def sync_command(target: RemoteTarget, tree: Path) -> list[str]:
    # The working tree, uncommitted edits included: the point is running what you are editing.
    # The remote path is re-split by the host's shell, so it is quoted; the remote dir is
    # created by the same rsync call rather than a separate ssh round trip.
    remote_dir = shlex.quote(target.dir)
    return [
        "rsync",
        "--archive",
        "--delete",
        *SYNC_FILTERS,
        f"--rsync-path=mkdir -p {remote_dir} && rsync",
        f"{tree}/",
        f"{target.ssh_target}:{remote_dir}/",
    ]


def run_command(target: RemoteTarget, argv: Sequence[str]) -> list[str]:
    remote = f"cd {shlex.quote(target.dir)} && exec uv run uni {shlex.join(argv)}"
    return ["ssh", target.ssh_target, remote]


def run_remote(target: RemoteTarget, argv: Sequence[str], tree: Path) -> int:
    """Sync `tree` to the host and run `uni argv` there, streaming its output here.

    Returns the exit code of the first step that fails, else the remote command's.
    """
    for command in (sync_command(target, tree), run_command(target, argv)):
        # [LAW:no-silent-failure] ssh and rsync speak for themselves on stderr; stop at the first miss.
        code = subprocess.run(command).returncode
        if code:
            return code
    return 0
