"""The repo is public and the run host is private: no tracked file may name the host."""

import re
import subprocess
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]

# What a host's identity looks like when it leaks into text. An ssh target's user is not
# preceded by a path separator, which keeps `owner/action@ref` pins out of it.
IDENTITY_PATTERNS = {
    "ssh target": re.compile(r"(?<![\w./-])[\w.-]+@[\w.-]+\b"),
    ".local hostname": re.compile(r"\b[\w-]+\.local\b"),
    "private IPv4": re.compile(
        r"(?<![\w.])(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}(?:\.\d{1,3})?(?![\w.])"
    ),
}


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True, text=True).stdout
    return [path for path in out.split("\0") if path]


def tracked_text() -> dict[str, str]:
    return {path: (ROOT / path).read_text(errors="replace") for path in tracked_files()}


def test_tracked_files_carry_no_host_identity():
    leaks = [
        (path, kind, match.group())
        for path, text in tracked_text().items()
        for kind, pattern in IDENTITY_PATTERNS.items()
        for match in pattern.finditer(text)
    ]
    assert leaks == []


def test_tracked_files_carry_no_value_from_the_real_env():
    # The patterns approximate an identity; your own .env defines it. Without a .env this checks nothing.
    secrets = [re.compile(rf"(?<![\w-]){re.escape(value)}(?![\w-])") for value in dotenv_values(ROOT / ".env").values() if value]
    leaks = [(path, secret.pattern) for path, text in tracked_text().items() for secret in secrets if secret.search(text)]
    assert leaks == []


def test_env_is_ignored_and_example_is_tracked():
    tracked = tracked_files()
    assert ".env" not in tracked
    assert ".env.example" in tracked
