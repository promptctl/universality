import pytest

from uni.pinned import PinnedConfigError, load_pinned, parse_pinned

VALID = """
[model]
id = "org/model"
revision = "7ae557604adf67be50417f59c2c2f167def9a775"
dtype = "float32"

[generation]
max_new_tokens = 8
"""


def test_committed_config_parses():
    assert load_pinned().revision


def test_valid_config_becomes_a_pinned_value():
    pinned = parse_pinned(VALID)
    assert (pinned.model_id, pinned.dtype, pinned.max_new_tokens) == ("org/model", "float32", 8)


@pytest.mark.parametrize(
    "old, new, field",
    [
        ('revision = "7ae557604adf67be50417f59c2c2f167def9a775"', 'revision = "main"', "model.revision"),
        ('dtype = "float32"', 'dtype = "float"', "model.dtype"),
        ("max_new_tokens = 8", "max_new_tokens = 0", "generation.max_new_tokens"),
        ("max_new_tokens = 8", "max_new_tokens = true", "generation.max_new_tokens must be a int"),
        ('revision = "7ae557604adf67be50417f59c2c2f167def9a775"', "revision = 123", "model.revision must be a str"),
        ('id = "org/model"', "", "model.id is missing"),
        ("[generation]\nmax_new_tokens = 8", "", "generation.max_new_tokens is missing"),
        ("[generation]\nmax_new_tokens = 8", 'generation = "max_new_tokens"', "generation.max_new_tokens is missing"),
    ],
)
def test_config_that_does_not_pin_is_refused(old, new, field):
    with pytest.raises(PinnedConfigError, match=field):
        parse_pinned(VALID.replace(old, new))
