"""CLI wiring: exit codes, GitHub annotations, dry-run plumbing. The pipeline itself is tested in test_workflow.py."""
from unittest.mock import MagicMock

import pytest

from src import main as cli
from src.pipeline import RunStats
from src.publishing.telegram import TelegramError


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for var in ("PUBLISH_MODE", "ITEMS_PER_SOURCE", "DRY_RUN", "GITHUB_STEP_SUMMARY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "@c")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("TELEGRAPH_ACCESS_TOKEN", "tok")


@pytest.fixture
def pipeline(monkeypatch):
    """Replace Pipeline with a recorder so no source, LLM or Telegram code runs."""
    created = {}

    class Recorder:
        def __init__(self, store, telegram, cfg, *, sources=()):
            created.update(store=store, telegram=telegram, cfg=cfg, sources=list(sources))

        def run(self):
            return created.pop("result", RunStats(summarized=2, published=2))

        def digest(self):
            return RunStats(digested=3)

    monkeypatch.setattr(cli, "Pipeline", Recorder)
    return created


def test_run_wires_real_sources_a_store_and_a_telegram_client(pipeline, capsys):
    assert cli.main(["run"]) == 0
    assert [type(s).__name__ for s in pipeline["sources"]] == ["HackerNewsSource", "GitHubTrendingSource", "ArxivSource"]
    assert pipeline["telegram"] is not None and pipeline["store"]._persist is True
    assert "run: fetched=0, selected=0, summarized=2, published=2" in capsys.readouterr().out


def test_digest_needs_no_sources(pipeline, capsys):
    assert cli.main(["digest"]) == 0 and pipeline["sources"] == []
    assert "digested=3" in capsys.readouterr().out


def test_dry_run_flag_means_no_telegram_client_and_no_persistence(pipeline):
    assert cli.main(["run", "--dry-run"]) == 0
    assert pipeline["telegram"] is None and pipeline["cfg"].dry_run is True and pipeline["store"]._persist is False


def test_dry_run_from_the_environment_works_too(pipeline, monkeypatch):
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN")  # a dry run must not need Telegram credentials
    assert cli.main(["run"]) == 0 and pipeline["telegram"] is None


def test_stopping_early_is_a_warning_not_a_failure(pipeline, capsys):
    pipeline["result"] = RunStats(summarized=1, stopped="Gemini quota exhausted (daily)")
    assert cli.main(["run"]) == 0
    assert "::warning title=Stopped early::Gemini quota exhausted (daily)" in capsys.readouterr().out


def test_job_summary_is_written_when_github_provides_the_file(pipeline, monkeypatch, tmp_path):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    cli.main(["run"])
    assert "summarized=2" in summary.read_text(encoding="utf-8")


@pytest.mark.parametrize("env, code", [({"PUBLISH_MODE": "both"}, 2), ({"ITEMS_PER_SOURCE": "x"}, 2)])
def test_invalid_configuration_exits_2_with_an_error_annotation(pipeline, monkeypatch, capsys, env, code):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert cli.main(["run"]) == code
    assert "::error title=Invalid configuration::" in capsys.readouterr().out


@pytest.mark.parametrize("missing, command, dry", [
    ("TELEGRAM_CHAT_ID", "run", False),
    ("GEMINI_API_KEY", "run", False),
    ("GEMINI_API_KEY", "run", True),            # a dry run still calls Gemini
    ("TELEGRAPH_ACCESS_TOKEN", "run", False),
    ("TELEGRAM_BOT_TOKEN", "digest", False),
])
def test_missing_secrets_fail_fast_naming_the_secret(pipeline, monkeypatch, capsys, missing, command, dry):
    monkeypatch.delenv(missing)
    assert cli.main([command] + (["--dry-run"] if dry else [])) == 2
    out = capsys.readouterr().out
    assert "::error title=Invalid configuration::" in out and missing in out
    assert "store" not in pipeline  # failed before the pipeline was even constructed: no quota spent


def test_secrets_that_a_command_does_not_need_are_not_required(pipeline, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("TELEGRAPH_ACCESS_TOKEN")
    assert cli.main(["digest"]) == 0                         # the digest never calls Gemini or Telegra.ph
    monkeypatch.setenv("WRITE_ARTICLES", "false")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert cli.main(["run"]) == 0                            # articles off: no Telegra.ph token needed


def test_configuration_errors_raised_mid_run_also_exit_2(monkeypatch, capsys):
    from src.llm.client import LLMConfigError
    from src.publishing.telegraph import TelegraphError

    for error in (LLMConfigError("Gemini API error 404 NOT_FOUND: models/gemini-1.5-flash is not found"),
                  TelegraphError("createPage: ACCESS_TOKEN_INVALID")):
        class Exploding:
            def __init__(self, *a, **k): ...
            def run(self, error=error):
                raise error
        monkeypatch.setattr(cli, "Pipeline", Exploding)
        assert cli.main(["run"]) == 2
    out = capsys.readouterr().out
    assert "gemini-1.5-flash" in out and "ACCESS_TOKEN_INVALID" in out


def test_corrupt_state_exits_3_and_is_not_reset(pipeline, tmp_path, capsys):
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "news.json").write_text("{broken")
    assert cli.main(["run"]) == 3
    assert (tmp_path / "state" / "news.json").read_text() == "{broken"  # untouched
    assert "::error title=State file::" in capsys.readouterr().out


def test_fatal_telegram_errors_exit_4(monkeypatch, capsys):
    class Exploding:
        def __init__(self, *a, **k): ...
        def run(self):
            raise TelegramError("Forbidden: bot was kicked", error_code=403)

    monkeypatch.setattr(cli, "Pipeline", Exploding)
    assert cli.main(["run"]) == 4
    assert "::error title=Telegram::Forbidden" in capsys.readouterr().out


@pytest.mark.parametrize("can_post, code", [(True, 0), (False, 4)])
def test_check_command_reports_posting_rights(monkeypatch, capsys, can_post, code):
    fake = MagicMock()
    fake.check.return_value = {"bot": "newsbot", "chat": "Tech News", "chat_type": "channel", "status": "member" if not can_post else "administrator", "can_post": can_post}
    monkeypatch.setattr(cli.TelegramClient, "from_env", classmethod(lambda cls, **k: fake))
    assert cli.main(["check"]) == code
    out = capsys.readouterr().out
    assert "newsbot" in out and ("OK: the bot can post" in out if can_post else "::error title=Telegram::" in out)


def test_source_selection_is_configurable(monkeypatch):
    monkeypatch.setenv("SOURCES", "arxiv")
    monkeypatch.setenv("ARXIV_KEYWORDS", "agent")
    sources = cli.build_sources(cli.Config.from_env())
    assert [type(s).__name__ for s in sources] == ["ArxivSource"] and sources[0]._keywords == ("agent",)
