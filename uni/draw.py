"""A picture as a file on disk. The one place this program knows what anything looks like."""

from __future__ import annotations

import io
from pathlib import Path

from uni.atomic import write_whole
from uni.figure import Picture

DPI = 160  # a figure a person looks at, at a size that shows a cascade's fourth split

# Small enough that a hundred thousand of them are a shape rather than a smear, which is the
# whole of what makes an orbit diagram readable: the branches are thin and they must stay thin.
DOT = 0.35


def scatter(picture: Picture, path: Path) -> Path:
    """Draw `picture` at `path`. One function, because both pictures are one kind of thing.

    A return map and an orbit diagram differ in what their points mean and in nothing about how
    they are drawn, so there is no second function here to keep in step with this one.
    [LAW:composability]
    """
    # Imported here rather than at the top: this module is the only one that needs a plotting
    # library, and `uni host` or a refused command must not pay for one.
    import matplotlib

    # Chosen before pyplot is imported, because pyplot picks a backend on import: the run host is
    # reached over ssh with no display, and a figure written to a file never wants a window.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(8, 6))
    try:
        drawn = axes.scatter(
            [point.x for point in picture.points],
            [point.y for point in picture.points],
            c=[point.shade for point in picture.points],
            s=DOT,
            cmap="viridis",
            linewidths=0,
        )
        # [LAW:dataflow-not-control-flow] the bar is keyed to a shade that means something, and a
        # picture whose points are all one shade says so by carrying no label for it.
        if picture.shade_label is not None:
            figure.colorbar(drawn, ax=axes, label=picture.shade_label)
        axes.set_title(picture.title)
        axes.set_xlabel(picture.x_label)
        axes.set_ylabel(picture.y_label)
        # Rendered into memory and handed to the one writer, rather than saved straight to the
        # path: a figure killed mid-write would otherwise leave a stump at the name this command
        # has already printed as the answer, in a directory that gets committed. `uni.atomic` says
        # every file this program writes goes through it, and a picture is a file.
        drawing = io.BytesIO()
        figure.savefig(drawing, format="png", dpi=DPI, bbox_inches="tight")
    finally:
        # Figures are held by pyplot until closed, so a command drawing several would grow without
        # this even though nothing here keeps a reference.
        plt.close(figure)
    return write_whole(path, drawing.getvalue())
