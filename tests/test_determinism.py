"""The determinism gate on the device named by --device (default: the pinned one).

Every run of a case must hash the same. The negative test at the bottom shows the failure the
gate exists for: the same prompt at batch size two does not reproduce batch size one.
"""

import subprocess
import sys

import pytest
import torch

from uni.determinism import ROOM, RUNS, cases, hashes

PROMPT = "Reply with one word: hello."
NEIGHBOR = "Write a long story about a lighthouse keeper who finds a message in a bottle on the shore."


@pytest.fixture(scope="module")
def gate(model):
    return {case.name: case for case in cases(model)}


@pytest.mark.parametrize("name", ["ordinary", "empty", "context-limit"])
def test_every_run_hashes_the_same(model, gate, name):
    assert len(set(hashes(model, gate[name], RUNS))) == 1


def test_a_fresh_process_hashes_the_same(model, gate):
    # Runs above share one process; thread counts and kernel choices are fixed per process.
    code = (
        "import sys; from dataclasses import replace; from uni.model import Model; from uni.pinned import load_pinned; "
        "print(Model(replace(load_pinned(), device=sys.argv[1])).generate(sys.argv[2]).sha256)"
    )
    prompt = gate["ordinary"].prompt
    fresh = subprocess.run([sys.executable, "-c", code, model.pinned.device, prompt], capture_output=True, text=True, check=True)
    assert fresh.stdout.split()[-1] == model.generate(prompt).sha256


def test_context_limit_case_generates_up_to_the_limit_and_no_further(model, gate):
    generation = model.generate(gate["context-limit"].prompt)
    assert len(generation.token_ids) == ROOM or generation.token_ids[-1] in model.stop_ids


def test_batch_size_two_does_not_reproduce_batch_size_one(model):
    """The prompt shares its batch with a longer request, left-padded to match, as a server batches.

    Greedy text happens to survive here, but the log-probabilities move around the fifth decimal
    (measured on mps and cpu). A near-tie anywhere in a long feedback loop flips a token, and one
    flipped token turns an orbit into noise. That is why generation is batch size one by construction.
    """
    alone = model.encode(PROMPT)
    neighbor = model.encode(NEIGHBOR)
    pad = neighbor.shape[1] - alone.shape[1]
    padded = torch.nn.functional.pad(alone, (pad, 0), value=model.tokenizer.pad_token_id)
    mask = torch.ones(2, neighbor.shape[1], dtype=torch.long, device=model.device)
    mask[0, :pad] = 0
    # Positions counted from the mask, so the padded prompt's tokens sit where they sit alone.
    # Without this the model numbers padding too and the test measures a position shift instead.
    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    with torch.inference_mode():
        one = model.model(input_ids=alone, logits_to_keep=1).logits[0, -1]
        two = model.model(input_ids=torch.cat([padded, neighbor]), attention_mask=mask, position_ids=positions, logits_to_keep=1).logits[0, -1]
    one, two = torch.log_softmax(one.float(), dim=-1), torch.log_softmax(two.float(), dim=-1)
    assert one.isfinite().all() and two.isfinite().all()  # a NaN would be unequal for the wrong reason
    assert not torch.equal(one, two)
