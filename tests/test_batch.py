"""Readings taken many rows to a forward pass, against the same readings taken a pass at a time, on each pinned model.

What the Determinism section records for `uni batch` is held here at a smaller grid: batch size one
through the batched pass is the reading alone to the bit, so what the larger sizes differ by is the
pass and not the arithmetic around it; no row's reading depends on the rows beside it; and no batch
moves a reading by more than a bound the recorded differences sit well inside.
"""

import pytest

from uni.batch import measure
from uni.model import Model
from uni.pinned import load_pinned
from uni.steer import Steer, read_direction
from uni.sweep import grid
from uni.template import load_templates

TEXT = "The meeting moved to Thursday because the room was booked."

# Twice the largest the section records on this repo's machines, 5.05e-05 in the answer and 9.02e-05 in
# a log-probability, so another Mac's kernels have room and a batch that reads a different model does not.
ANSWER_BOUND = 1e-4
LOGPROB_BOUND = 2e-4

# Each model's committed response curve: the direction it pushes along, the layer the answer is read at,
# and pushes across the range its hump is read over. 16 rows reach the size at which a kernel changes.
CURVES = {
    "qwen2.5-0.5b": ("formality", 23, "-10:10:16"),
    "smollm2-360m": ("formality-16", 28, "-3:3:16"),
}


@pytest.fixture(scope="module", params=list(CURVES))
def measured(request):
    name = request.param
    pinned = load_pinned(name)
    # The session's model when it is the default, so the suite loads each checkpoint once.
    model = request.getfixturevalue("model") if pinned == load_pinned() else Model(pinned)
    knob, layer, pushes = CURVES[name]
    prompt = load_templates()["rewrite"].render(TEXT)
    return measure(model, prompt, Steer(read_direction(knob, pinned)), grid(pushes), layer, (1, 2, 3, 16))


def test_one_row_to_a_batched_pass_is_the_reading_alone_to_the_bit(measured):
    one = measured.sizes[0]
    assert one.size == 1 and one.from_alone.answer == 0 and one.from_alone.logprob == 0


def test_a_rows_reading_does_not_depend_on_the_rows_beside_it(measured):
    assert measured.neighbours.answer == 0 and measured.neighbours.logprob == 0


def test_no_batch_moves_a_reading_past_the_bound(measured):
    for size in measured.sizes:
        assert size.from_alone.answer <= ANSWER_BOUND and size.from_alone.logprob <= LOGPROB_BOUND, size
