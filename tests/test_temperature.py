"""What one token drawn after the prompt puts into the response map's answer, on the pinned model."""

import json
from pathlib import Path

import pytest
import torch

from uni.curve import read_curve
from uni.model import CANDIDATES
from uni.response import response
from uni.steer import Steer, read_direction
from uni.temperature import Draw, Drawn, drawn
from uni.template import load_templates

PROMPT = load_templates()["rewrite"].render("The meeting moved to Thursday because the room was booked.")
TEXT = "The meeting moved to Thursday because the room was booked."


@pytest.fixture(scope="module")
def formality(model):
    return Steer(read_direction("formality", model.pinned))


@pytest.fixture(scope="module")
def read(model, formality):
    # One pass over the whole vocabulary, kept with what the model handed `drawn`, so the tests below
    # can check both against the model read another way without paying for the pass again.
    seen = []
    original = model.next_tokens
    model.next_tokens = lambda *args: seen.append(original(*args)) or seen[-1]
    try:
        found = drawn(model, PROMPT, formality, 0.0, 23, (1.0, 0.05))
    finally:
        del model.next_tokens
    return found, seen[0]


def appended(model, additions, token):
    """The stream leaving layer 23 at the prompt's tokens and `token` after them, read in one pass with nothing kept between."""
    ids = torch.cat([model.encode(PROMPT), torch.tensor([[token]], device=model.device)], dim=1)
    captured = []
    with model._residual(additions), torch.inference_mode():
        handle = model.layers[23].register_forward_hook(lambda _m, _i, hidden: captured.append(hidden))
        try:
            model.model(input_ids=ids, logits_to_keep=1)
        finally:
            handle.remove()
    return captured[0][0]


def test_with_no_token_the_answer_is_the_response(model, formality, read):
    found, _ = read
    assert found.response == pytest.approx(response(model, PROMPT, formality, 0.0, 23), abs=1e-4)


def test_every_candidate_is_read_as_if_it_had_been_appended_whichever_pass_it_fell_in(model, formality, read):
    _, (_, logprobs, own) = read
    turned = formality.turn(0.0)
    pushed = sum(model.residual_vector(addition) for addition in turned.additions)
    assert logprobs.shape == own.shape == (model.model.config.vocab_size,)
    assert float(logprobs.exp().sum()) == pytest.approx(1.0, abs=1e-9)
    for token in (0, CANDIDATES - 1, CANDIDATES, len(own) - 1):
        stream = appended(model, turned.additions, token)
        assert float(own[token]) == pytest.approx(formality.direction.project(stream[-1] - pushed), abs=1e-3)


def test_cold_the_draw_is_the_likeliest_token_and_hot_it_spreads(model, formality, read):
    found, (_, logprobs, _) = read
    hot, cold = found.draws
    turned = formality.turn(0.0)
    pushed = sum(model.residual_vector(addition) for addition in turned.additions)
    # At 0.05 the likeliest token outweighs the next by e^(20 times their gap in log-probability).
    greedy = appended(model, turned.additions, int(torch.argmax(logprobs)))
    assert (hot.temperature, cold.temperature) == (1.0, 0.05)
    assert cold.mean == pytest.approx(formality.direction.project((greedy - pushed).mean(dim=0)), abs=1e-3)
    assert cold.spread < 1e-6 < 1e-2 < hot.spread


def test_a_run_refuses_a_push_before_it_reads_any(model, monkeypatch, capsys, tmp_path):
    from uni.cli import EXIT_CONFIG, main

    monkeypatch.setattr("uni.model.Model", lambda pinned: model)
    monkeypatch.setattr("uni.temperature.drawn", lambda *_: pytest.fail("read a push of a run that was going to be refused"))
    argv = ["temperature", "--template", "rewrite", "--knob", "formality", "--start", TEXT, "--grid", "0:1e12:2", "--layer", "23", "--temperature", "1", "--curves", str(tmp_path)]
    assert main(argv, {}, Path.cwd()) == EXIT_CONFIG
    assert "a push this large drowns out what the model writes" in capsys.readouterr().err


def test_a_run_writes_each_temperature_s_means_and_spreads_as_curves(model, monkeypatch, capsys, tmp_path):
    from uni.cli import main

    monkeypatch.setattr("uni.model.Model", lambda pinned: model)
    monkeypatch.setattr("uni.temperature.drawn", lambda model, prompt, steer, value, layer, temperatures: Drawn(value, tuple(Draw(t, value + t, value * t) for t in temperatures)))
    argv = ["temperature", "--template", "rewrite", "--knob", "formality", "--start", TEXT, "--grid", "1:2:2", "--layer", "23", "--temperature", "0.5", "--temperature", "2", "--curves", str(tmp_path)]
    assert main(argv, {}, Path.cwd()) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed[0].split() == ["value", "no", "token", "mean", "at", "0.5", "spread", "at", "0.5", "mean", "at", "2", "spread", "at", "2"]
    assert [line.split()[:4] for line in printed[1:3]] == [["1", "1.0000", "1.5000", "5.0000e-01"], ["2", "2.0000", "2.5000", "1.0000e+00"]]
    for line, temperature in zip(printed[3:], (0.5, 2.0)):
        means, spreads = (Path(part.split(" in ")[1].rstrip(",")) for part in line.split(": ")[1].split(", spreads"))
        assert read_curve(means, 23).readings == (1 + temperature, 2 + temperature)
        assert read_curve(spreads, 23).readings == (temperature, 2 * temperature)
        described = json.loads(spreads.read_text())["described"]
        assert (described["temperature"], described["reading"], described["start"]) == (temperature, "spread", TEXT)


def test_temperature_is_refused_on_the_host_whose_curves_would_not_come_back(capsys):
    from uni.cli import EXIT_CONFIG, main

    argv = ["--remote", "temperature", "--template", "rewrite", "--knob", "formality", "--start", "a", "--grid", "0:0:1", "--layer", "23", "--temperature", "1"]
    assert main(argv, {}, Path.cwd()) == EXIT_CONFIG
    assert "temperature writes a file into this checkout, so it runs here, not on the host" in capsys.readouterr().err
