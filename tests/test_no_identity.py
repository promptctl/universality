"""The repo is public and the run host is private: no tracked file may name the host."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# What a host's identity looks like when it leaks into text.
IDENTITY_PATTERNS = {
    "ssh target": re.compile(r"\b[\w.-]+@[\w.-]+\b"),
    ".local hostname": re.compile(r"\b[\w-]+\.local\b"),
    "private IPv4": re.compile(r"\b(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}(?:\.\d{1,3})?\b"),
}


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True, text=True).stdout
    return [path for path in out.split("\0") if path]


def test_tracked_files_carry_no_host_identity():
    leaks = [
        (path, kind, match.group())
        for path in tracked_files()
        for kind, pattern in IDENTITY_PATTERNS.items()
        for match in pattern.finditer((ROOT / path).read_text(errors="replace"))
    ]
    assert leaks == []


def test_env_is_ignored_and_example_is_tracked():
    tracked = tracked_files()
    assert ".env" not in tracked
    assert ".env.example" in tracked
