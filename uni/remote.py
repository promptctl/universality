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

# Never carried to the host: the venv is rebuilt there, .git is not needed to run,
# and .env holds the host's own identity.
SYNC_EXCLUDES = (".git", ".venv", ".env", "__pycache__", ".pytest_cache")


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
            f"{', '.join(missing)} not set; copy .env.example to .env and fill it in"
        )
    host, user, dir = (env[name] for name in VARIABLES)
    if not dir.startswith("/"):
        raise RemoteConfigError(f"UNI_REMOTE_DIR must be an absolute path on the host, got {dir!r}")
    return RemoteTarget(host=host, user=user, dir=dir)


# [LAW:effects-at-boundaries] the three steps are pure descriptions; run_remote performs them.


def make_dir_command(target: RemoteTarget) -> list[str]:
    return ["ssh", target.ssh_target, f"mkdir -p {shlex.quote(target.dir)}"]


def sync_command(target: RemoteTarget, tree: Path) -> list[str]:
    # The working tree, uncommitted edits included: the point is running what you are editing.
    return [
        "rsync",
        "--archive",
        "--delete",
        *(f"--exclude={name}" for name in SYNC_EXCLUDES),
        f"{tree}/",
        f"{target.ssh_target}:{target.dir}/",
    ]


def run_command(target: RemoteTarget, argv: Sequence[str]) -> list[str]:
    remote = f"cd {shlex.quote(target.dir)} && exec uv run uni {shlex.join(argv)}"
    return ["ssh", target.ssh_target, remote]


def run_remote(target: RemoteTarget, argv: Sequence[str], tree: Path) -> int:
    """Sync `tree` to the host and run `uni argv` there, streaming its output here.

    Returns the exit code of the first step that fails, else the remote command's.
    """
    for command in (make_dir_command(target), sync_command(target, tree), run_command(target, argv)):
        # [LAW:no-silent-failure] ssh and rsync speak for themselves on stderr; stop at the first miss.
        code = subprocess.run(command).returncode
        if code:
            return code
    return 0
