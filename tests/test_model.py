"""The wrapper on the pinned model and device. The first run downloads the checkpoint."""

import pytest
import torch

from uni.model import Model, ResidualAdd
from uni.pinned import load_pinned

PROMPT = "Reply with one word: hello."


@pytest.fixture(scope="module")
def model() -> Model:
    return Model(load_pinned())


@pytest.fixture(scope="module")
def baseline(model):
    return model.generate(PROMPT)


def test_model_runs_on_the_pinned_device(model):
    assert next(model.model.parameters()).device.type == torch.device(load_pinned().device).type


def test_generation_carries_one_logprob_per_token(model, baseline):
    assert baseline.text
    assert len(baseline.token_ids) == len(baseline.tokens) == len(baseline.logprobs) > 0
    assert all(logprob <= 0 for logprob in baseline.logprobs)
    assert baseline.token_ids[-1] in model.stop_ids


def test_zero_vector_on_the_residual_stream_changes_nothing(model, baseline):
    zero = ResidualAdd(layer=len(model.layers) // 2, vector=torch.zeros(model.hidden_size))
    assert model.generate(PROMPT, [zero]) == baseline


def test_large_vector_on_the_residual_stream_changes_the_output(model, baseline):
    large = ResidualAdd(layer=len(model.layers) // 2, vector=torch.full((model.hidden_size,), 1000.0))
    assert model.generate(PROMPT, [large]).text != baseline.text


def test_hooks_are_removed_after_generation(model, baseline):
    large = ResidualAdd(layer=len(model.layers) // 2, vector=torch.full((model.hidden_size,), 1000.0))
    model.generate(PROMPT, [large])
    assert model.generate(PROMPT) == baseline


@pytest.mark.parametrize(
    "layer, size, message",
    [(-1, None, "layer must be in"), (10_000, None, "layer must be in"), (0, 3, r"vector must have shape")],
)
def test_malformed_additions_are_refused_before_generating(model, layer, size, message):
    addition = ResidualAdd(layer=layer, vector=torch.zeros(size or model.hidden_size))
    with pytest.raises(ValueError, match=message):
        model.generate(PROMPT, [addition])
