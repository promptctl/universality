"""A file this program writes is whole or it is not there at all."""

from __future__ import annotations

import contextlib
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
    thing renaming into place exists to make impossible.

    [LAW:single-enforcer] every file this program writes is written here, so the rule cannot hold
    for trajectories and not for the manifest that names them.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Appended to the whole name rather than put in place of the extension, so the scratch file is
    # one file's and not one stem's: a directory holding both `a.json` and `a.txt` would otherwise
    # have them writing through each other, which is the collision above with the pid left out.
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    try:
        with partial.open("wb") as file:
            file.write(data)
            file.flush()
            # On the disk before the rename, and not merely in the page cache: a rename can be
            # durable while the bytes under it are not, and this program reads "the file is
            # there" as "that work is done". A power cut between the two would leave a finished
            # cell holding nothing - and nothing looks at it again, because looking again is
            # exactly what the name being there rules out. One disk write per file, against an
            # orbit that cost minutes to produce. [LAW:no-silent-failure]
            os.fsync(file.fileno())
        partial.replace(path)
    finally:
        # Nothing left behind by a failure this process can see: a Ctrl-C, a full disk. A signal
        # that ends the process outright runs no finally block and does leave its scratch file,
        # which is what the process id in the name is for - that litter is inert, and the run
        # resuming the work makes its own.
        #
        # Suppressed because a cleanup that raises replaces the failure it was cleaning up after:
        # where the rename failed because the directory went read-only, the unlink fails for the
        # same reason, and the user would read an error about the scratch file instead of about
        # the write. A scratch file left behind is the inert litter above; a lost error is not.
        with contextlib.suppress(OSError):
            partial.unlink(missing_ok=True)
    return path
