"""Run a `uni` command on the experiment host.

The host's identity never lives in this tree. It arrives through the environment,
normally a gitignored `.env` (see `.env.example`), and is parsed once here into a
`RemoteTarget` that the rest of the program takes as a plain value.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from uni.parse import ConfigError

VARIABLES = ("UNI_REMOTE_HOST", "UNI_REMOTE_USER", "UNI_REMOTE_DIR")

# Plain characters only, so the path needs no quoting on either side of ssh: rsync
# versions disagree about whether the remote shell re-splits it.
REMOTE_DIR = re.compile(r"/[\w./-]+")

# What git ignores stays home (rsync reads .gitignore itself; negated patterns are not
# understood). .git is not needed to run and .env holds the host's identity. .venv, trajectories/,
# sweeps/ and figures/ are the host's own tools and results, and they are named here as plain
# excludes rather than left to the gitignore filter: the sync runs with --delete, and what keeps
# the host's results out of its reach should not depend on a per-directory .gitignore being found
# and read the same way at both ends. figures/ is the one of them that is committed, and it is
# excluded for the same reason rather than in spite of it: nothing on the host reads a figure, and
# under --delete a local figures/ would delete a picture the host had just spent a GPU pass
# drawing. Each machine keeps its own orbits, sweeps and pictures.
SYNC_FILTERS = (
    "--exclude=.git",
    "--exclude=.env",
    "--exclude=.venv",
    "--exclude=/trajectories/",
    "--exclude=/sweeps/",
    "--exclude=/figures/",
    "--filter=:- .gitignore",
)


class RemoteConfigError(ConfigError):
    """The environment does not describe a usable host. The message says what to fix.

    A ConfigError because that is what it is - the run as described cannot be run - and because
    the CLI answered it exactly like one anyway, from a clause of its own. Two types with one
    behaviour is a distinction that does nothing. [LAW:one-type-per-behavior]
    """


@dataclass(frozen=True)
class RemoteTarget:
    host: str
    user: str
    dir: str  # absolute path on the host, matching REMOTE_DIR

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
    if not REMOTE_DIR.fullmatch(dir):
        raise RemoteConfigError(
            f"UNI_REMOTE_DIR must be an absolute path of letters, digits, '.', '_', '-' and '/', got {dir!r}"
        )
    return RemoteTarget(host=host, user=user, dir=dir)


# [LAW:effects-at-boundaries] the two steps are pure descriptions; run_remote performs them.


def sync_command(target: RemoteTarget, tree: Path) -> list[str]:
    # The working tree, uncommitted edits included: the point is running what you are editing.
    # The remote dir is created by the same rsync call rather than a separate ssh round trip.
    return [
        "rsync",
        "--archive",
        "--delete",
        *SYNC_FILTERS,
        f"--rsync-path=mkdir -p {target.dir} && rsync",
        f"{tree}/",
        f"{target.ssh_target}:{target.dir}/",
    ]


def run_command(target: RemoteTarget, argv: Sequence[str]) -> list[str]:
    remote = f"cd {target.dir} && exec uv run uni {shlex.join(argv)}"
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
