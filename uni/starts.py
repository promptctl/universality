"""Start sets: named families of start states, parsed once from starts.toml."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from importlib.resources import files
from typing import Any

from uni.parse import ConfigError, field


class StartsError(ConfigError):
    """starts.toml does not hold start sets, or a run named one it does not hold. The message says which."""


def prefixes(name: str, raw: Mapping[str, Any]) -> tuple[str, ...]:
    """The set's passage cut after each of its word counts, in the order the counts are written.

    A family with one parameter, the length of the start, so that a sweep over it spreads its starts
    across the observable a return map is read along - which one start, or a handful typed by hand,
    cannot do. Whether two counts name one start is the sweep's to refuse, as it refuses a `--start`
    given twice. [LAW:single-enforcer]
    """
    if not isinstance(raw, Mapping):
        raise StartsError(f"start set {name!r} must be a table with a passage and its word counts")
    try:
        words = field(raw, "passage", str, StartsError).split()
        counts = field(raw, "words", list, StartsError)
    except StartsError as error:
        raise StartsError(f"start set {name!r}: {error}") from error
    # [LAW:no-silent-failure] a count past the end would cut nothing and hand back the whole passage
    # under a number that says otherwise.
    for count in counts:
        if type(count) is not int or not 1 <= count <= len(words):
            raise StartsError(f"start set {name!r}: word counts are whole numbers from 1 to the passage's {len(words)}, got {count!r}")
    return tuple(" ".join(words[:count]) for count in counts)


def load_starts() -> dict[str, tuple[str, ...]]:
    try:
        raw = tomllib.loads(files("uni").joinpath("starts.toml").read_text())
    except tomllib.TOMLDecodeError as error:
        raise StartsError(f"uni/starts.toml is not TOML: {error}") from error
    return {name: prefixes(name, table) for name, table in raw.items()}


def named_starts(names: Sequence[str]) -> tuple[str, ...]:
    """Every start of the sets named, set after set, refused unless starts.toml holds each one."""
    if not names:
        return ()  # a sweep given only --start reads no file
    sets = load_starts()
    unknown = [name for name in names if name not in sets]
    if unknown:
        raise StartsError(f"no start set {', '.join(map(repr, unknown))}; the start sets are {', '.join(sets)}")
    return tuple(start for name in names for start in sets[name])
