"""The repo is public and the run host is private: no tracked file may name the host."""

import re
import subprocess
from pathlib import Path

import pytest
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]

# The length at which finding a value inside a compressed image stops being a coincidence: six
# bytes is about one chance in forty thousand across ten megabytes of them, where three bytes is
# about one in forty. Below it, a value is looked for in the text files alone.
BINARY_FLOOR = 6

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


def tracked_text(raw: dict[str, bytes] | None = None) -> dict[str, str]:
    """The tracked files that are text, as text. A file that is not text is not in here at all.

    Handed the bytes when the caller already has them, so a check that wants both reads the repo
    once rather than twice. Reading them itself is the convenience for a caller that wants only
    the text.

    Decoded strictly, and not with `errors="replace"`: a figure forced through a text decode comes
    out as mojibake in which `IDENTITY_PATTERNS` finds an ssh target every few thousand bytes, so
    committing one picture would fail this file with a hundred leaks that are not there. The shapes
    below describe text, so text is what they are read over - and the literal check that follows
    reads every tracked file as bytes, which is where a real value hiding in a binary would be
    caught. [LAW:parse-dont-validate]
    """
    text = {}
    for path, content in (raw if raw is not None else tracked_bytes()).items():
        try:
            decoded = content.decode()
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


def value_leaks(values: list[str], text: dict[str, str], binary: dict[str, bytes]) -> list[tuple[str, str]]:
    """Where any of these values sits whole: in every text file, and in a binary one only when the
    value is long enough that finding it there is the value rather than the entropy."""
    found = []
    for value in values:
        secret = re.compile(rf"(?<![\w-]){re.escape(value)}(?![\w-])".encode())
        searched = {path: raw.encode() for path, raw in text.items()}
        if len(value) >= BINARY_FLOOR:
            searched |= binary
        found += [(path, value) for path, raw in searched.items() if secret.search(raw)]
    return found


@pytest.mark.parametrize(
    "value, found_in",
    [
        ("shortie", ["figure.png", "notes.md"]),  # long enough that a figure is worth searching
        ("abc", ["notes.md"]),  # too short: a hit in compressed bytes would be chance, not a leak
    ],
)
def test_a_short_value_is_looked_for_where_a_person_could_have_written_it_and_nowhere_else(value, found_in):
    text = {"notes.md": f"the box is {value} today"}
    binary = {"figure.png": b"\x89PNG\r\n\x1a\n" + value.encode() + b"\x00\xff"}
    assert sorted(path for path, _ in value_leaks([value], text, binary)) == found_in


def test_tracked_files_carry_no_value_from_the_real_env():
    # The patterns approximate an identity; your own .env defines it. Without a .env this checks nothing.
    # Searched as bytes over every tracked file, text or not - the figures the scan above skips
    # included, which is the only check that reads them at all, and which reads nothing without a
    # .env of your own.
    #
    # Short values are searched in the text files only. A literal is a literal wherever it sits,
    # but a few bytes of compressed image are a few random bytes: across the pictures committed
    # here a three-byte value has a percent or so of matching nothing, and that failure would read
    # as a host name leaked into a figure. Past BINARY_FLOOR the arithmetic settles it - a
    # six-byte value has about one chance in forty thousand across ten megabytes - so the line is
    # drawn where chance stops being the likelier explanation, and what it costs is named: a value
    # shorter than that is checked everywhere a person could have written it and nowhere else.
    values = [value for value in dotenv_values(ROOT / ".env").values() if value]
    # Read once each. `tracked_text` reads every tracked file itself when it has to, so asking it
    # inside the comprehension below read the whole repo once for every file in it - and asking it
    # beside `tracked_bytes` still read everything twice. It takes the bytes now.
    every = tracked_bytes()
    text = tracked_text(every)
    binary = {path: content for path, content in every.items() if path not in text}
    assert value_leaks(values, text, binary) == []


def test_env_is_ignored_and_example_is_tracked():
    tracked = tracked_files()
    assert ".env" not in tracked
    assert ".env.example" in tracked
