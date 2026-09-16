"""A file this program writes is whole or it is not there at all."""

from __future__ import annotations

import os
from pathlib import Path


def write_whole(path: Path, data: bytes) -> Path:
    """Put `data` at `path`, leaving whatever was there untouched unless all of it lands.

    Written beside the file and renamed over it, because rename is the one file operation that
    either happened or did not: a run killed mid-write leaves the earlier file whole, and no
    reader ever sees half of one.

    The scratch name carries this process's id because the commands here are meant to be rerun -
    resuming a sweep is running the same command again - so two of them reach the same file
    together, and under one scratch name the second would truncate the bytes the first was about
    to rename into place, promoting a half-written file to a finished one, which is the single
    thing renaming into place exists to make impossible. The cost is that a killed run leaves its
    scratch file behind instead of overwriting it next time, which is litter rather than damage.

    [LAW:single-enforcer] every file this program writes is written here, so the rule cannot hold
    for trajectories and not for the manifest that names them.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Appended to the whole name rather than put in place of the extension, so the scratch file is
    # one file's and not one stem's: a directory holding both `a.json` and `a.txt` would otherwise
    # have them writing through each other, which is the collision above with the pid left out.
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    partial.write_bytes(data)
    partial.replace(path)
    return path
