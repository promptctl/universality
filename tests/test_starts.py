"""Start sets: a passage cut at word counts, named on the command line instead of typed into it."""

from pathlib import Path

import pytest

from uni.cli import EXIT_CONFIG, main
from uni.starts import StartsError, load_starts, named_starts, prefixes

PASSAGE = "one two  three\nfour five"


def test_a_set_is_its_passage_cut_after_each_word_count_in_order():
    assert prefixes("s", {"passage": PASSAGE, "words": [3, 1, 5]}) == ("one two three", "one", "one two three four five")


@pytest.mark.parametrize("count", [0, 6, 2.0, True])
def test_a_word_count_the_passage_cannot_be_cut_at_is_refused(count):
    # 6 is the silent one: cut past the end, the passage comes back whole under a count that says otherwise.
    with pytest.raises(StartsError, match="whole numbers from 1 to the passage's 5"):
        prefixes("s", {"passage": PASSAGE, "words": [count]})


@pytest.mark.parametrize("raw, missing", [({"words": [1]}, "passage"), ({"passage": PASSAGE}, "words")])
def test_a_set_missing_a_field_is_refused_by_name(raw, missing):
    with pytest.raises(StartsError, match=f"start set 's': {missing} is missing"):
        prefixes("s", raw)


def test_the_shipped_sets_parse_and_spread_over_length():
    sets = load_starts()
    assert {"harbor", "memo"} <= sets.keys()
    for starts in sets.values():
        lengths = [len(start) for start in starts]
        assert lengths == sorted(lengths) and lengths[0] < 10 < 500 < lengths[-1]


def test_naming_no_set_reads_no_file(monkeypatch):
    monkeypatch.setattr("uni.starts.load_starts", lambda: pytest.fail("read starts.toml for no set"))
    assert named_starts([]) == ()


def status(tmp_path, monkeypatch, capsys, *starts):
    monkeypatch.setattr("uni.cli.SWEEPS", tmp_path)
    argv = ["sweep", "--template", "rewrite", "--grid", "0:0:1", "--steps", "1", "--status", *starts]
    code = main(argv, {}, Path.cwd())
    return code, capsys.readouterr()


def test_a_named_set_is_the_same_sweep_as_its_starts_typed_after_any_start(tmp_path, monkeypatch, capsys):
    # The sweep's name is the hash of its starts, so equal names are equal starts in equal order.
    typed = ["--start", "first"] + [arg for start in load_starts()["memo"] for arg in ("--start", start)]
    _, by_hand = status(tmp_path, monkeypatch, capsys, *typed)
    code, by_name = status(tmp_path, monkeypatch, capsys, "--starts", "memo", "--start", "first")
    assert code == 0
    assert by_name.out == by_hand.out
    assert f"0 of {1 + len(load_starts()['memo'])} cells done" in by_name.out


def test_an_unknown_set_is_refused_with_the_sets_there_are(tmp_path, monkeypatch, capsys):
    code, printed = status(tmp_path, monkeypatch, capsys, "--starts", "nope")
    assert code == EXIT_CONFIG
    assert "no start set 'nope'; the start sets are harbor, memo" in printed.err


def test_a_sweep_given_no_starts_at_all_is_refused(tmp_path, monkeypatch, capsys):
    code, printed = status(tmp_path, monkeypatch, capsys)
    assert code == EXIT_CONFIG
    assert "needs at least one of starts" in printed.err
