import pytest

from src.config import Config, ConfigError


def test_defaults_when_nothing_is_set():
    cfg = Config.from_env({})
    assert (cfg.publish_mode, cfg.items_per_source, cfg.write_articles, cfg.dry_run) == ("single", 2, True, False)
    assert cfg.sources == ("hackernews", "github", "arxiv")
    assert cfg.arxiv_keywords == ("LLM", "agent", "reasoning", "RAG", "transformer")  # conservative by default


def test_arxiv_keywords_can_be_turned_off_explicitly():
    assert Config.from_env({"ARXIV_KEYWORDS": "none"}).arxiv_keywords == ()
    assert Config.from_env({"ARXIV_KEYWORDS": "OFF"}).arxiv_keywords == ()
    # an empty string still means "use the default", because GitHub passes unset variables as ""
    assert Config.from_env({"ARXIV_KEYWORDS": ""}).arxiv_keywords == Config.from_env({}).arxiv_keywords


def test_github_passes_unset_variables_as_empty_strings_and_they_mean_default():
    env = {name: "" for name in (
        "PUBLISH_MODE", "ITEMS_PER_SOURCE", "MAX_POSTS_PER_RUN", "WRITE_ARTICLES", "MAX_RUNTIME_SECONDS", "DIGEST_SIZE",
        "DIGEST_MAX_AGE_HOURS", "STATE_DIR", "SOURCES", "HN_MIN_POINTS", "ARXIV_CATEGORIES", "ARXIV_KEYWORDS", "DRY_RUN")}
    assert Config.from_env(env) == Config.from_env({})
    assert Config.from_env(env).arxiv_keywords == Config.from_env({}).arxiv_keywords
    assert Config.from_env(env) == Config.from_env({})
    assert Config.from_env({"ITEMS_PER_SOURCE": "   "}).items_per_source == 2  # whitespace too


def test_values_are_parsed():
    cfg = Config.from_env({
        "PUBLISH_MODE": "DIGEST", "ITEMS_PER_SOURCE": "3", "WRITE_ARTICLES": "no", "DRY_RUN": "True",
        "SOURCES": "hackernews, arxiv", "ARXIV_KEYWORDS": "agent, LLM ,", "HN_MIN_POINTS": "0", "STATE_DIR": "/tmp/s",
    })
    assert cfg.publish_mode == "digest" and cfg.items_per_source == 3 and cfg.write_articles is False and cfg.dry_run is True
    assert cfg.sources == ("hackernews", "arxiv") and cfg.arxiv_keywords == ("agent", "LLM")
    assert cfg.hn_min_points == 0 and cfg.state_dir == "/tmp/s"


@pytest.mark.parametrize("env", [
    {"PUBLISH_MODE": "both"},
    {"ITEMS_PER_SOURCE": "two"},
    {"ITEMS_PER_SOURCE": "0"},
    {"MAX_RUNTIME_SECONDS": "5"},
    {"WRITE_ARTICLES": "maybe"},
    {"SOURCES": "hackernews,reddit"},
])
def test_invalid_values_fail_early_with_the_variable_name(env):
    with pytest.raises(ConfigError, match=next(iter(env)) if "SOURCES" not in env else "SOURCES"):
        Config.from_env(env)


def test_modules_that_read_env_directly_also_treat_empty_as_unset(monkeypatch):
    from src.llm import client
    from src.publishing import templates

    monkeypatch.setenv("GEMINI_MODEL", "")
    monkeypatch.setenv("CHANNEL_HANDLE", "")
    templates.default_branding.cache_clear()
    assert templates.default_branding().channel_handle == "@your_channel"
    monkeypatch.setenv("CHANNEL_HANDLE", "@real_channel")
    templates.default_branding.cache_clear()
    assert templates.default_branding().channel_handle == "@real_channel"
    templates.default_branding.cache_clear()

    # the model name is resolved inside call_model; prove an empty variable falls back to the default
    from unittest.mock import MagicMock, patch
    from google.genai import types
    sdk = MagicMock()
    sdk.models.generate_content.return_value = types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[types.Part(text="{}")]), finish_reason=types.FinishReason.STOP)])
    with patch.object(client, "_client", return_value=sdk):
        client.call_model([{"role": "user", "content": "x"}], {"json_schema": {"schema": {}}}, max_completion_tokens=10)
    assert sdk.models.generate_content.call_args.kwargs["model"] == client.DEFAULT_MODEL

    # USER_AGENT is read at import time: check in a fresh interpreter instead of reloading the module here
    import os, subprocess, sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, "-c", "import src.ingestion.base as b; print(b.USER_AGENT)"],
                         cwd=root, env={**os.environ, "BOT_USER_AGENT": ""}, capture_output=True, text=True, check=True)
    assert out.stdout.startswith("TechNewsBot/")
