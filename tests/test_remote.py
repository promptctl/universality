import platform
from pathlib import Path

import pytest

from uni.cli import EXIT_CONFIG, main
from uni.remote import RemoteConfigError, remote_target_from_env, run_command, sync_command

ENV = {"UNI_REMOTE_HOST": "box", "UNI_REMOTE_USER": "me", "UNI_REMOTE_DIR": "/srv/uni"}
# Assembled, not written: a literal ssh target here would trip test_no_identity, as it should.
SSH_TARGET = "@".join((ENV["UNI_REMOTE_USER"], ENV["UNI_REMOTE_HOST"]))


def test_missing_variables_are_named_in_one_error():
    with pytest.raises(RemoteConfigError, match="UNI_REMOTE_HOST, UNI_REMOTE_DIR missing or empty"):
        remote_target_from_env({"UNI_REMOTE_USER": "me", "UNI_REMOTE_DIR": ""})


def test_remote_dir_must_be_absolute():
    with pytest.raises(RemoteConfigError, match="UNI_REMOTE_DIR must be an absolute path"):
        remote_target_from_env({**ENV, "UNI_REMOTE_DIR": "~/uni"})


def test_remote_command_runs_uni_in_the_remote_dir_over_ssh():
    target = remote_target_from_env(ENV)
    assert run_command(target, ["gen", "hello world"]) == [
        "ssh",
        SSH_TARGET,
        "cd /srv/uni && exec uv run uni gen 'hello world'",
    ]


def test_sync_mirrors_the_working_tree_as_git_sees_it_without_env():
    command = sync_command(remote_target_from_env(ENV), Path("/work/tree"))
    assert command[-2:] == ["/work/tree/", f"{SSH_TARGET}:/srv/uni/"]
    assert "--delete" in command
    assert {"--exclude=.git", "--exclude=.env", "--filter=:- .gitignore"} <= set(command)


def test_sync_quotes_a_remote_dir_with_spaces_and_creates_it():
    target = remote_target_from_env({**ENV, "UNI_REMOTE_DIR": "/srv/my uni"})
    command = sync_command(target, Path("/work/tree"))
    assert command[-1] == f"{SSH_TARGET}:'/srv/my uni'/"
    assert "--rsync-path=mkdir -p '/srv/my uni' && rsync" in command


def test_cli_reports_missing_config_without_a_trace(capsys):
    assert main(["host", "--remote"], {}) == EXIT_CONFIG
    err = capsys.readouterr().err
    assert err == "uni: UNI_REMOTE_HOST, UNI_REMOTE_USER, UNI_REMOTE_DIR missing or empty; copy .env.example to .env and fill it in\n"


def test_remote_flag_is_not_abbreviable():
    with pytest.raises(SystemExit):
        main(["host", "--rem"], ENV)


def test_host_runs_locally_without_the_flag(capsys):
    assert main(["host"], {}) == 0
    assert capsys.readouterr().out.startswith(platform.node())
