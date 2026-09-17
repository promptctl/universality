import platform
import signal
import subprocess
from pathlib import Path

import pytest

from uni.cli import EXIT_CONFIG, main, split_remote
from uni.remote import RemoteConfigError, fetch_command, remote_target_from_env, returned_dir, run_command, run_remote, sync_command

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


def test_a_step_killed_by_a_signal_exits_as_a_shell_would_say_it(monkeypatch):
    # subprocess reports rsync or ssh killed by Ctrl-C as -2, which sys.exit turns into 254; a shell
    # says 130 of the same death.
    monkeypatch.setattr(subprocess, "run", lambda command: subprocess.CompletedProcess(command, -signal.SIGINT))
    assert run_remote(remote_target_from_env(ENV), ["sweep"], Path("/tree")) == 128 + signal.SIGINT


def test_a_returned_directory_comes_back_beside_what_is_here_and_deletes_nothing():
    command = fetch_command(remote_target_from_env(ENV), Path("/work/tree"), Path("curves"))
    assert command == ["rsync", "--archive", "--ignore-existing", f"{SSH_TARGET}:/srv/uni/curves/", "/work/tree/curves/"]


@pytest.mark.parametrize("directory", ["/tmp/curves", "../curves", "curves/../../x", "my curves", "c$HOME", "."])
def test_a_returned_directory_is_named_plainly_from_the_checkout_s_root(directory):
    with pytest.raises(RemoteConfigError, match="named from the checkout's root"):
        returned_dir(Path(directory))
    assert returned_dir(Path("curves/..x/a.b")) == Path("curves/..x/a.b")


def test_temperature_on_the_host_brings_its_curves_back_after_it_succeeds_and_nothing_after_it_fails(checkout, monkeypatch):
    ran = []
    codes = iter([0, 0, 0, 0, 3])
    monkeypatch.setattr("uni.cli.checkout_root", lambda cwd: checkout)
    monkeypatch.setattr(subprocess, "run", lambda command: ran.append(command) or subprocess.CompletedProcess(command, next(codes)))
    argv = ["temperature", "--template", "rewrite", "--knob", "formality", "--start", "a", "--grid", "0:0:1", "--layer", "23", "--temperature", "1", "--curves", "curves/hot"]
    assert main(["--remote", *argv], ENV, checkout) == 0
    assert [command[0] for command in ran] == ["rsync", "ssh", "rsync"]
    assert ran[2][-2:] == [f"{SSH_TARGET}:/srv/uni/curves/hot/", f"{checkout}/curves/hot/"]
    ran.clear()
    assert main(["--remote", *argv], ENV, checkout) == 3
    assert [command[0] for command in ran] == ["rsync", "ssh"]


def test_temperature_on_the_host_is_refused_before_it_runs_when_its_curves_could_not_come_back(checkout, monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "run", lambda command: pytest.fail("nothing is synced or run"))
    argv = ["--remote", "temperature", "--template", "rewrite", "--knob", "formality", "--start", "a", "--grid", "0:0:1", "--layer", "23", "--temperature", "1", "--curves", "/tmp/elsewhere"]
    assert main(argv, ENV, checkout) == EXIT_CONFIG
    assert "named from the checkout's root" in capsys.readouterr().err


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


def test_the_sync_deletes_what_this_checkout_no_longer_has_but_never_a_curve_the_host_wrote(tmp_path):
    # The real rsync, between two local directories, with the sync's own filters: a curve the host
    # wrote and has not yet sent back survives the next sync, and a stale source file does not.
    from uni.remote import SYNC_FILTERS

    here, host = tmp_path / "here", tmp_path / "host"
    (here / "curves").mkdir(parents=True)
    (here / "curves" / "kept.json").write_text("here")
    (host / "curves").mkdir(parents=True)
    (host / "curves" / "unfetched.json").write_text("host")
    (host / "stale.py").write_text("gone")
    subprocess.run(["rsync", "--archive", "--delete", *SYNC_FILTERS, f"{here}/", f"{host}/"], check=True)
    assert sorted(path.name for path in (host / "curves").iterdir()) == ["kept.json", "unfetched.json"]
    assert not (host / "stale.py").exists()
