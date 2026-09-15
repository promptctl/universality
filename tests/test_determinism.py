"""The determinism gate on the device named by --device (default: the pinned one).

Every run of a case must hash the same. The negative test at the bottom shows the failure the
gate exists for: the same prompt at batch size two does not reproduce batch size one.
"""

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
    with torch.inference_mode():
        one = model.model(input_ids=alone, logits_to_keep=1).logits[0, -1]
        two = model.model(input_ids=torch.cat([padded, neighbor]), attention_mask=mask, logits_to_keep=1).logits[0, -1]
    assert not torch.equal(torch.log_softmax(one.float(), dim=-1), torch.log_softmax(two.float(), dim=-1))
