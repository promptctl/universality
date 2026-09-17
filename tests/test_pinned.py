from pathlib import Path

import pytest

from uni.cli import EXIT_CONFIG, main
from uni.pinned import PinnedConfigError, load_pinned, parse_pinned

VALID = """
default = "small.v2"

[models."small.v2"]
id = "org/model"
revision = "7ae557604adf67be50417f59c2c2f167def9a775"
dtype = "float32"

[models.other]
id = "org/other"
revision = "a10cc1512eabd3dde888204e902eca88bddb4951"
dtype = "bfloat16"

[generation]
max_new_tokens = 8
"""


def test_committed_config_parses_and_names_its_default_and_the_second_model():
    assert load_pinned().model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert load_pinned("qwen2.5-0.5b") == load_pinned()
    assert load_pinned("smollm2-360m").model_id == "HuggingFaceTB/SmolLM2-360M-Instruct"


def test_valid_config_becomes_the_pinned_value_it_names_and_its_default_without_a_name():
    pinned = parse_pinned(VALID)
    assert (pinned.model_id, pinned.dtype, pinned.max_new_tokens) == ("org/model", "float32", 8)
    other = parse_pinned(VALID, "other")
    assert (other.model_id, other.dtype, other.max_new_tokens) == ("org/other", "bfloat16", 8)
    # Named by what it is, so the name a table goes by is not part of what a trajectory records.
    assert set(vars(other)) == {"model_id", "revision", "dtype", "max_new_tokens"}
    assert other.home == "org--other"


@pytest.mark.parametrize(
    "old, new, field",
    [
        ('revision = "7ae557604adf67be50417f59c2c2f167def9a775"', 'revision = "main"', "models.small.v2.revision must be a full commit sha"),
        ('dtype = "float32"', 'dtype = "float"', "models.small.v2.dtype"),
        ("max_new_tokens = 8", "max_new_tokens = 0", "generation.max_new_tokens"),
        ("max_new_tokens = 8", "max_new_tokens = true", "generation.max_new_tokens must be a int"),
        ('revision = "7ae557604adf67be50417f59c2c2f167def9a775"', "revision = 123", "models.small.v2.revision must be a str"),
        ('id = "org/model"', "", "models.small.v2.id is missing"),
        ("[generation]\nmax_new_tokens = 8", "", "generation.max_new_tokens is missing"),
        ("[generation]\nmax_new_tokens = 8", 'generation = "max_new_tokens"', "generation.max_new_tokens is missing"),
        ('default = "small.v2"', 'default = "large"', "no model 'large' is pinned; the models are small.v2, other"),
        ('default = "small.v2"', "", "default is missing"),
    ],
)
def test_config_that_does_not_pin_is_refused(old, new, field):
    with pytest.raises(PinnedConfigError, match=field.replace(".", "\\.").replace("(", "\\(")):
        parse_pinned(VALID.replace(old, new, 1))


def test_a_model_that_is_not_pinned_is_refused_by_name():
    with pytest.raises(PinnedConfigError, match="no model 'huge' is pinned; the models are small.v2, other"):
        parse_pinned(VALID, "huge")
    with pytest.raises(PinnedConfigError, match="models must hold a table for each pinned model"):
        parse_pinned('default = "x"\n[generation]\nmax_new_tokens = 8\n')


def test_the_command_reports_a_bad_pin_rather_than_a_traceback(capsys, monkeypatch):
    def refuse(name=None):
        raise PinnedConfigError("models.qwen2.5-0.5b.dtype must be one of float32, float16, bfloat16")

    # Every command reads the pin inside itself, long after argparse has had its say.
    monkeypatch.setattr("uni.pinned.load_pinned", refuse)
    assert main(["gen", "hi"], {}, Path.cwd()) == EXIT_CONFIG
    assert "uni: models.qwen2.5-0.5b.dtype must be one of" in capsys.readouterr().err


def test_a_model_nobody_pinned_is_refused_before_a_checkpoint_is_read(capsys):
    assert main(["gen", "hi", "--model", "huge"], {}, Path.cwd()) == EXIT_CONFIG
    assert "no model 'huge' is pinned; the models are qwen2.5-0.5b, smollm2-360m" in capsys.readouterr().err


def test_one_model_id_is_pinned_once_since_it_names_the_directory_its_directions_are_kept_in():
    twice = VALID.replace('id = "org/other"', 'id = "org/model"')
    with pytest.raises(PinnedConfigError, match="org/model is pinned more than once"):
        parse_pinned(twice)
