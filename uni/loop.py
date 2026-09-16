"""The loop runner: iterate any Map from a start state, and keep the orbit as a file."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from uni.parse import ConfigError, field


class Map(Protocol):
    """One step of an iterated map. States are text whatever the map, so every brick reads every orbit."""

    @property
    def spec(self) -> Mapping[str, Any]:
        """Everything besides the start that fixes the orbit, as JSON data."""
        ...

    def step(self, state: str) -> str: ...


def orbit(map: Map, start: str) -> Iterator[str]:
    """The states after each step, without end; the caller takes as many as it wants."""
    # [LAW:composability] the runner knows only the Map protocol, never which map it iterates.
    state = start
    while True:
        state = map.step(state)
        yield state


class TrajectoryError(ConfigError):
    """A file does not hold a trajectory. The message says which field is wrong."""


@dataclass(frozen=True)
class Trajectory:
    map: Mapping[str, Any]
    value: float  # the knob's setting, which the map bakes in; recorded so a sweep reads it without decoding the knob
    start: str
    states: tuple[str, ...]  # the state after each step, so step n is states[n - 1]

    @property
    def name(self) -> str:
        """The file name, from what fixes the orbit, so rerunning a command rewrites its own file."""
        inputs = json.dumps([self.map, self.value, self.start, len(self.states)], sort_keys=True)
        return hashlib.sha256(inputs.encode()).hexdigest()[:16] + ".json"

    def encode(self) -> bytes:
        # Sorted keys and no timestamp: the same orbit is the same bytes.
        fields = {"map": self.map, "value": self.value, "start": self.start, "states": list(self.states)}
        return (json.dumps(fields, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def write_trajectory(trajectory: Trajectory, dir: Path) -> Path:
    dir.mkdir(parents=True, exist_ok=True)
    path = dir / trajectory.name
    # Written beside it and renamed over it, so a rerun killed mid-write leaves the earlier file whole.
    partial = path.with_suffix(".partial")
    partial.write_bytes(trajectory.encode())
    partial.replace(path)
    return path


def _field(raw: dict[str, Any], key: str, kind: type) -> Any:
    return field(raw, key, kind, TrajectoryError)


def read_trajectory(path: Path) -> Trajectory:
    # [LAW:parse-dont-validate] a file becomes a Trajectory here or not at all.
    try:
        raw = json.loads(path.read_bytes())
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
