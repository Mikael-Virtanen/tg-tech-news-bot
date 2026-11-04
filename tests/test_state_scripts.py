"""
The state scripts run for real, against a local bare git repository standing in for GitHub.
They are the only place where a bug could silently destroy the dedup memory, so they get real git,
not mocks.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is required")


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def remote(tmp_path):
    path = tmp_path / "origin.git"
    git("init", "--bare", "-b", "main", str(path))
    return path


@pytest.fixture
def env(tmp_path, remote):
    return {
        **os.environ,
        "STATE_REMOTE": f"file://{remote}",
        "STATE_DIR": str(tmp_path / "work" / ".state"),
        "GIT_CONFIG_GLOBAL": "/dev/null",  # keep the developer's own git config out of the test
        "GIT_TERMINAL_PROMPT": "0",
    }


def script(name, env, cwd=None):
    return subprocess.run(["bash", str(SCRIPTS / name)], env=env, cwd=cwd, capture_output=True, text=True)


def commit_count(remote, branch="state"):
    return int(git("--git-dir", str(remote), "rev-list", "--count", branch))


def test_first_run_starts_empty_then_saves_and_restores(env, remote):
    state = Path(env["STATE_DIR"])

    pulled = script("state_pull.sh", env)
    assert pulled.returncode == 0 and "starting with an empty state" in pulled.stdout and state.is_dir()

    (state / "news.json").write_text('{"version": 1, "items": {"a:1": {}}}', encoding="utf-8")
    pushed = script("state_push.sh", env)
    assert pushed.returncode == 0, pushed.stderr
    assert commit_count(remote) == 1

    shutil.rmtree(state)  # a brand new runner: nothing on disk
    assert script("state_pull.sh", env).returncode == 0
    assert (state / "news.json").read_text(encoding="utf-8") == '{"version": 1, "items": {"a:1": {}}}'


def test_every_push_replaces_history_so_the_repository_never_grows(env, remote):
    state = Path(env["STATE_DIR"])
    script("state_pull.sh", env)
    for n in range(4):
        (state / "news.json").write_text(f'{{"version": 1, "items": {{"run": {n}}}}}', encoding="utf-8")
        assert script("state_push.sh", env).returncode == 0
        script("state_pull.sh", env)  # each workflow run starts from a fresh clone
    assert commit_count(remote) == 1
    shown = git("--git-dir", str(remote), "show", "state:news.json")
    assert '"run": 3' in shown


def test_unchanged_state_is_not_pushed(env, remote):
    state = Path(env["STATE_DIR"])
    script("state_pull.sh", env)
    (state / "news.json").write_text("{}", encoding="utf-8")
    script("state_push.sh", env)
    first_sha = git("--git-dir", str(remote), "rev-parse", "state")

    script("state_pull.sh", env)
    again = script("state_push.sh", env)
    assert "unchanged" in again.stdout
    assert git("--git-dir", str(remote), "rev-parse", "state") == first_sha


def test_an_empty_state_directory_pushes_nothing(env, remote):
    script("state_pull.sh", env)
    result = script("state_push.sh", env)
    assert result.returncode == 0 and "nothing to save" in result.stdout
    assert subprocess.run(["git", "--git-dir", str(remote), "rev-parse", "state"], capture_output=True).returncode != 0


def test_an_unreachable_remote_is_NOT_mistaken_for_a_first_run(env, tmp_path):
    """The dangerous case: treating an outage as 'no state yet' would let the next push wipe the real state."""
    env = {**env, "STATE_REMOTE": f"file://{tmp_path / 'does-not-exist.git'}"}
    result = script("state_pull.sh", env)
    assert result.returncode == 1
    assert "refusing to continue" in result.stderr
    assert not Path(env["STATE_DIR"]).exists()  # no empty directory left behind that a push could publish


def test_missing_credentials_fail_loudly(env):
    stripped = {k: v for k, v in env.items() if k not in ("STATE_REMOTE", "GITHUB_TOKEN", "GITHUB_REPOSITORY")}
    assert script("state_pull.sh", stripped).returncode != 0


# ------------------------------ keep-alive ----------------------------------- #
@pytest.fixture
def clone(tmp_path, remote):
    work = tmp_path / "clone"
    git("clone", "--quiet", f"file://{remote}", str(work))
    git("config", "user.name", "t", cwd=work)
    git("config", "user.email", "t@example.com", cwd=work)
    return work


def commit_at(work, when):
    (work / "f.txt").write_text(when)
    git("add", "-A", cwd=work)
    env = {**os.environ, "GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when}
    subprocess.run(["git", "commit", "-q", "-m", "work"], cwd=work, env=env, check=True)
    git("push", "-q", "origin", "HEAD:main", cwd=work)


def test_keepalive_adds_an_empty_commit_only_when_the_repo_has_gone_quiet(clone, env):
    commit_at(clone, "2020-01-01T00:00:00")
    stale = script("keepalive.sh", env, cwd=clone)
    assert stale.returncode == 0, stale.stderr
    assert "pushed an empty commit" in stale.stdout
    assert git("log", "-1", "--format=%s", cwd=clone) == "chore: keep scheduled workflows active"

    fresh = script("keepalive.sh", env, cwd=clone)  # the repo is now active again
    assert "nothing to do" in fresh.stdout
    assert git("rev-list", "--count", "HEAD", cwd=clone) == "2"
