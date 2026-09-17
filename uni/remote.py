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
# Where every file a command writes on the host comes back to. [LAW:one-source-of-truth] the sync
# protects this directory from --delete and the fetch accepts nothing outside it, so a result the
# host wrote and did not yet send back is never lost to the next sync, wherever it was written.
RETURNED_ROOT = "curves"
# A directory a command writes into: RETURNED_ROOT or one inside it, plain for the same reason, and
# never a climb out of it. A part that is `.` or `..` is refused: `.` alone would fetch the host's
# whole checkout back over this one.
RETURNED_DIR = re.compile(rf"(?!(?:.*/)?\.{{1,2}}(?:/|$)){RETURNED_ROOT}(?:/[\w.-]+)*")

# What git ignores stays home (rsync reads .gitignore itself; negated patterns are not
# understood). .git is not needed to run and .env holds the host's identity. .venv, trajectories/,
# sweeps/ and figures/ are the host's own tools and results, and they are named here as plain
# excludes rather than left to the gitignore filter: the sync runs with --delete, and what keeps
# the host's results out of its reach should not depend on a per-directory .gitignore being found
# and read the same way at both ends. figures/ is the one of them that is committed, and it is
# excluded for the same reason rather than in spite of it: nothing on the host reads a figure, and
# under --delete a local figures/ would delete a picture the host had just spent a GPU pass
# drawing. Each machine keeps its own orbits, sweeps and pictures.
#
# curves/, RETURNED_ROOT, is sent, because a map the host runs is fitted to a curve here, and it is
# protected from --delete, because the host writes curves into it too. A curve is named by its own bytes, so one
# the host has and this checkout lacks is never stale: it is a result whose fetch did not happen
# yet, and the next sync must not be what loses it.
SYNC_FILTERS = (
    "--exclude=.git",
    "--exclude=.env",
    "--exclude=.venv",
    "--exclude=/trajectories/",
    "--exclude=/sweeps/",
    "--exclude=/figures/",
    f"--filter=P /{RETURNED_ROOT}/**",
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


def returned_dir(directory: Path) -> Path:
    """A directory a command run on the host writes into, as one the fetch can name at both ends, or a refusal."""
    if not RETURNED_DIR.fullmatch(str(directory)):
        raise RemoteConfigError(
            f"a directory the host writes into comes back into the same place in this checkout, and only {RETURNED_ROOT}/ is kept from the next sync's deletions, so it is {RETURNED_ROOT} or a directory in it, named in letters, digits, '.', '_', '-' and '/', with no part that is '.' or '..'; got {str(directory)!r}"
        )
    return directory


def fetch_command(target: RemoteTarget, tree: Path, directory: Path) -> list[str]:
    # The files come back beside the ones already here and never over them, and nothing here is
    # deleted: what returns is named by its own content, so a name already here is those bytes.
    return [
        "rsync",
        "--archive",
        "--ignore-existing",
        f"{target.ssh_target}:{target.dir}/{directory}/",
        f"{tree}/{directory}/",
    ]


def run_remote(target: RemoteTarget, argv: Sequence[str], tree: Path, returned: Sequence[Path] = ()) -> int:
    """Sync `tree` to the host, run `uni argv` there streaming its output here, and bring back each of the `returned` directories it wrote into.

    Returns the exit code of the first step that fails, else the remote command's. A command that
    failed brings nothing back.
    """
    for command in (sync_command(target, tree), run_command(target, argv), *(fetch_command(target, tree, directory) for directory in returned)):
        # [LAW:no-silent-failure] ssh and rsync speak for themselves on stderr; stop at the first miss.
        returncode = subprocess.run(command).returncode
        # A step killed by a signal - rsync or ssh on a Ctrl-C, say - comes back as minus the
        # signal, which sys.exit would turn into 256 minus it. 128 plus it is what a shell says of
        # the same death, and the number every reader of an exit code already knows. (Not `| head`:
        # ssh ignores SIGPIPE, and a closed pipe reaches it as a write error it exits on itself.)
        code = returncode if returncode >= 0 else 128 - returncode
        if code:
            return code
    return 0
