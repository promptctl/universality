"""The loop runner: iterate any Map from a start state, and keep the orbit as a file."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from uni.atomic import write_whole
from uni.parse import ConfigError, field


class Map(Protocol):
    """One step of an iterated map. States are text whatever the map, so every brick reads every orbit."""

    @property
    def spec(self) -> Mapping[str, Any]:
        """Everything besides the start and the value that fixes the orbit, as JSON data."""
        ...

    @property
    def value(self) -> float:
        """The map's one scalar parameter: r, or the knob's setting. Recorded beside the spec.

        Asked of the map rather than read off the command line a second time, so what the file
        says the orbit ran at is what it ran at. [LAW:one-source-of-truth]
        """
        ...

    def step(self, state: str) -> str: ...


def orbit(map: Map, start: str) -> Iterator[str]:
    """The states after each step, without end; the caller takes as many as it wants."""
    # [LAW:composability] the runner knows only the Map protocol, never which map it iterates.
    state = start
    while True:
        state = map.step(state)
        yield state


def trajectory_name(map: Mapping[str, Any], value: float, start: str, steps: int) -> str:
    """The file an orbit of this many steps writes itself to, from what fixes it and nothing else.

    Known before the orbit is run, which is what lets a sweep ask whether a cell is already
    on disk without a second record of what it has done. [LAW:one-source-of-truth]
    """
    inputs = json.dumps([map, float(value), start, steps], sort_keys=True)
    return hashlib.sha256(inputs.encode()).hexdigest()[:16] + ".json"


class TrajectoryError(ConfigError):
    """A file does not hold a trajectory. The message says which field is wrong."""


@dataclass(frozen=True)
class Trajectory:
    map: Mapping[str, Any]
    value: float  # the knob's setting, which the map bakes in; recorded so a sweep reads it without decoding the knob
    start: str
    states: tuple[str, ...]  # the state after each step, so step n is states[n - 1]

    def __post_init__(self) -> None:
        # One number, one spelling, fixed here because this is what holds the file's shape rather
        # than in each map that hands one over. A `Logistic(3)` would otherwise write "value": 3,
        # which reads back as an int the parser refuses - and, because `name` hashes the value,
        # would name a second file for an orbit that already has one. [LAW:single-enforcer]
        object.__setattr__(self, "value", float(self.value))

    @property
    def name(self) -> str:
        """The file name, from what fixes the orbit, so rerunning a command rewrites its own file."""
        return trajectory_name(self.map, self.value, self.start, len(self.states))

    def encode(self) -> bytes:
        # Sorted keys and no timestamp: the same orbit is the same bytes.
        fields = {"map": self.map, "value": self.value, "start": self.start, "states": list(self.states)}
        return (json.dumps(fields, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def write_trajectory(trajectory: Trajectory, dir: Path) -> Path:
    # [LAW:no-silent-failure] whole or absent, which is what lets a sweep read "this cell is done"
    # off the directory listing: see uni.atomic for why that holds even under two runs at once.
    return write_whole(dir / trajectory.name, trajectory.encode())


def _field(raw: dict[str, Any], key: str, kind: type) -> Any:
    return field(raw, key, kind, TrajectoryError)


def read_trajectory(path: Path) -> Trajectory:
    # [LAW:parse-dont-validate] a file becomes a Trajectory here or not at all.
    try:
        raw = json.loads(path.read_bytes())
    except OSError as error:  # the path is one the user typed, so a mistyped one is theirs to fix, not a bug here
        raise TrajectoryError(f"{path} cannot be read: {error.strerror}") from error
    except json.JSONDecodeError as error:  # a run killed mid-write leaves exactly this
        raise TrajectoryError(f"{path} is not JSON: {error}") from error
    if type(raw) is not dict:
        raise TrajectoryError(f"{path} must hold a JSON object")
    states = _field(raw, "states", list)
    if not all(type(state) is str for state in states):
        raise TrajectoryError("states must all be strings")
    return Trajectory(
        map=_field(raw, "map", dict),
        value=_field(raw, "value", float),
        start=_field(raw, "start", str),
        states=tuple(states),
    )
