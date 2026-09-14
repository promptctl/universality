import pytest

from uni.pinned import PinnedConfigError, load_pinned, parse_pinned

VALID = """
[model]
id = "org/model"
revision = "7ae557604adf67be50417f59c2c2f167def9a775"
dtype = "float32"
device = "cpu"

[generation]
max_new_tokens = 8
"""


def test_committed_config_parses():
    assert load_pinned().revision


def test_valid_config_becomes_a_pinned_value():
    pinned = parse_pinned(VALID)
    assert (pinned.model_id, pinned.dtype, pinned.device, pinned.max_new_tokens) == ("org/model", "float32", "cpu", 8)


@pytest.mark.parametrize(
    "old, new, field",
    [
        ('revision = "7ae557604adf67be50417f59c2c2f167def9a775"', 'revision = "main"', "model.revision"),
        ('dtype = "float32"', 'dtype = "float"', "model.dtype"),
        ("max_new_tokens = 8", "max_new_tokens = 0", "generation.max_new_tokens"),
    ],
)
def test_config_that_does_not_pin_is_refused(old, new, field):
    with pytest.raises(PinnedConfigError, match=field):
        parse_pinned(VALID.replace(old, new))
