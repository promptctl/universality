import platform
import subprocess
from pathlib import Path

import pytest

from uni.cli import EXIT_CONFIG, main, split_remote
from uni.remote import RemoteConfigError, remote_target_from_env, run_command, sync_command

ENV = {"UNI_REMOTE_HOST": "box", "UNI_REMOTE_USER": "me", "UNI_REMOTE_DIR": "/srv/uni"}
# Assembled, not written: a literal ssh target here would trip test_no_identity, as it should.
SSH_TARGET = "@".join((ENV["UNI_REMOTE_USER"], ENV["UNI_REMOTE_HOST"]))


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return tmp_path


def test_missing_variables_are_named_in_one_error():
    with pytest.raises(RemoteConfigError, match="UNI_REMOTE_HOST, UNI_REMOTE_DIR missing or empty"):
        remote_target_from_env({"UNI_REMOTE_USER": "me", "UNI_REMOTE_DIR": ""})


@pytest.mark.parametrize("dir", ["~/uni", "/srv/my uni", "/srv/$HOME"])
def test_remote_dir_must_be_absolute_and_plain(dir):
    with pytest.raises(RemoteConfigError, match="UNI_REMOTE_DIR must be an absolute path"):
        remote_target_from_env({**ENV, "UNI_REMOTE_DIR": dir})


def test_remote_command_runs_uni_in_the_remote_dir_over_ssh():
    target = remote_target_from_env(ENV)
    assert run_command(target, ["gen", "hello world"]) == [
        "ssh",
        SSH_TARGET,
        "cd /srv/uni && exec uv run uni gen 'hello world'",
    ]


def test_sync_mirrors_the_working_tree_as_git_sees_it_and_creates_the_dir():
    command = sync_command(remote_target_from_env(ENV), Path("/work/tree"))
    assert command[-2:] == ["/work/tree/", f"{SSH_TARGET}:/srv/uni/"]
    assert "--delete" in command
    assert "--rsync-path=mkdir -p /srv/uni && rsync" in command
    assert {"--exclude=.git", "--exclude=.env", "--exclude=.venv", "--exclude=/trajectories/", "--filter=:- .gitignore"} <= set(command)


def test_split_remote_forwards_everything_else_verbatim():
    assert split_remote(["host", "--remote"]) == (True, ["host"])
    assert split_remote(["gen", "--remote", "--", "--remote"]) == (True, ["gen", "--", "--remote"])
    assert split_remote(["gen", "--rem", "x"]) == (False, ["gen", "--rem", "x"])


def test_cli_reports_missing_config_without_a_trace(checkout, capsys):
    assert main(["host", "--remote"], {}, checkout) == EXIT_CONFIG
    err = capsys.readouterr().err
    assert err == "uni: UNI_REMOTE_HOST, UNI_REMOTE_USER, UNI_REMOTE_DIR missing or empty; copy .env.example to .env and fill it in\n"


def test_cli_refuses_remote_outside_a_checkout(tmp_path, capsys):
    assert main(["host", "--remote"], ENV, tmp_path) == EXIT_CONFIG
    assert capsys.readouterr().err.startswith("uni: --remote runs from inside a checkout")


def test_remote_flag_is_not_abbreviable(tmp_path):
    with pytest.raises(SystemExit):
        main(["host", "--rem"], ENV, tmp_path)


def test_host_runs_locally_without_the_flag(tmp_path, capsys):
    assert main(["host"], {}, tmp_path) == 0
    assert capsys.readouterr().out.startswith(platform.node())
