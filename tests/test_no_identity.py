"""The repo is public and the run host is private: no tracked file may name the host."""

import re
import subprocess
from pathlib import Path

import pytest
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]

# What a host's identity looks like when it leaks into text. Public shapes that share the
# surface, an action pinned to a tag or sha, a version string, a file suffix, are carved out
# by what follows, so a URL-form ssh target still counts. uv.lock writes four-part versions in
# two places: after `version = "`, and in a distribution filename, followed by `-py`.
IDENTITY_PATTERNS = {
    "ssh target": re.compile(r"(?<![\w.-])[\w.-]+@(?!v?\d)(?![0-9a-f]{7,40}\b)[\w-]+(?:\.[\w-]+)*"),
    ".local hostname": re.compile(r"(?<![\w.])[\w-]+\.local(?!\.?\w)"),
    "private IPv4": re.compile(r"(?<![\w.])(?<!version = \")(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}(?!\.?\w|-py)"),
}


def leaks_in(text: str) -> list[tuple[str, str]]:
    return [(kind, m.group()) for kind, pattern in IDENTITY_PATTERNS.items() for m in pattern.finditer(text)]


# Joined at test time: written whole, these lines would be leaks in a tracked file.
@pytest.mark.parametrize(
    "parts",
    [
        ("ssh me@", "box"),
        ("git clone ssh://me@", "box/srv/uni"),
        ("the box is inferno-two", ".local, up"),
        ("The run host is at 192.168", ".1.20."),
        ("reach 10.0", ".0.7 or 172.16", ".4.4"),
        ('host = "10.0', '.0.5"'),
        ("ping host-10.0", ".0.5"),
        ("scan 10.0", ".0.1-10.0", ".0.9"),
    ],
)
def test_identity_shapes_are_caught(parts):
    assert leaks_in("".join(parts))


@pytest.mark.parametrize(
    "text",
    [
        "uses: actions/checkout@3d3c42e5cd17d92e8be7bc3d47bd2ef2ef0d4d18",
        "uses: promptctl/copirate-code-review-agent@v1",
        "uv tool install ruff@0.6.9",
        "ignore .claude/settings.local.json and .env.local",
        "macOS 10.15.7 is the oldest; Python 3.10.0.1 too",
        'name = "nvidia-curand"\nversion = "10.4.0.35"',
        "files.pythonhosted.org/packages/1e/72/nvidia_curand-10.4.0.35-py3-none-manylinux_2_27_aarch64.whl",
    ],
)
def test_public_shapes_are_not_leaks(text):
    assert leaks_in(text) == []


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True, text=True).stdout
    return [path for path in out.split("\0") if path]


def tracked_bytes() -> dict[str, bytes]:
    return {path: (ROOT / path).read_bytes() for path in tracked_files()}


def tracked_text() -> dict[str, str]:
    """The tracked files that are text, as text. A file that is not text is not in here at all.

    Decoded strictly, and not with `errors="replace"`: a figure forced through a text decode comes
    out as mojibake in which `IDENTITY_PATTERNS` finds an ssh target every few thousand bytes, so
    committing one picture would fail this file with a hundred leaks that are not there. The shapes
    below describe text, so text is what they are read over - and the literal check that follows
    reads every tracked file as bytes, which is where a real value hiding in a binary would be
    caught. [LAW:parse-dont-validate]
    """
    text = {}
    for path, raw in tracked_bytes().items():
        try:
            decoded = raw.decode()
        except UnicodeDecodeError:
            continue
        if "\0" not in decoded:  # git's own tell, so a UTF-8-decodable binary is still binary
            text[path] = decoded
    return text


def test_tracked_files_carry_no_host_identity():
    assert [(path, *leak) for path, text in tracked_text().items() for leak in leaks_in(text)] == []


def test_what_the_text_scan_skips_is_exactly_the_committed_figures():
    # Skipping is how the guard stays about what it says it is about: compressed image bytes hold
    # user-at-host shapes by the hundred, so the first committed picture would otherwise turn it
    # off with ninety-six leaks that are not there. But a skip nothing names is a skip nobody
    # notices, so what got skipped is asserted rather than trusted - a new file this scan cannot
    # read fails here until someone has looked at it. [LAW:no-silent-failure]
    skipped = set(tracked_files()) - set(tracked_text())
    assert skipped == {path for path in tracked_files() if path.endswith(".png")}
    assert skipped <= set(tracked_bytes())  # and every one of them is still read for a real value


def test_tracked_files_carry_no_value_from_the_real_env():
    # The patterns approximate an identity; your own .env defines it. Without a .env this checks nothing.
    # Searched as bytes over every tracked file, text or not: a literal host name is a literal
    # host name wherever it sits, and unlike the shapes above it cannot turn up in a PNG by chance.
    secrets = [re.compile(rf"(?<![\w-]){re.escape(value)}(?![\w-])".encode()) for value in dotenv_values(ROOT / ".env").values() if value]
    leaks = [(path, secret.pattern) for path, raw in tracked_bytes().items() for secret in secrets if secret.search(raw)]
    assert leaks == []


def test_env_is_ignored_and_example_is_tracked():
    tracked = tracked_files()
    assert ".env" not in tracked
    assert ".env.example" in tracked
