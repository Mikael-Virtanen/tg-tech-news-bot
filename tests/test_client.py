"""Gemini client tests. No network or API key needed: the SDK client is replaced by a fake."""
import json
from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors, types

from src.llm import client, summarizer
from src.llm.prompts import RESPONSE_FORMAT, SYSTEM_PROMPT
from src.models import NewsItem

MESSAGES = [
    {"role": "system", "content": "SYS"},
    {"role": "user", "content": "first"},
    {"role": "assistant", "content": "draft"},
    {"role": "user", "content": "fix it"},
]


def gemini_response(text: str | None = '{"ok": true}', finish=types.FinishReason.STOP) -> types.GenerateContentResponse:
    parts = [types.Part(text=text)] if text is not None else []
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts), finish_reason=finish)]
    )


def quota_error(message="Quota exceeded.", delay="1s", quota_id="GenerateRequestsPerMinutePerProjectPerModel-FreeTier"):
    body = {"error": {"code": 429, "message": message, "status": "RESOURCE_EXHAUSTED", "details": [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay},
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": quota_id}]},
    ]}}
    return errors.ClientError(429, body)


@pytest.fixture
def fake(monkeypatch):
    """Patch the SDK client and clear env so each test controls its own configuration."""
    for var in ("GEMINI_MODEL", "GEMINI_THINKING_LEVEL", "GEMINI_TEMPERATURE"):
        monkeypatch.delenv(var, raising=False)
    sdk = MagicMock()
    sdk.models.generate_content.return_value = gemini_response()
    with patch.object(client, "_client", return_value=sdk), patch.object(client.time, "sleep") as sleep:
        sdk.sleep = sleep
        yield sdk


def call(**overrides):
    kwargs = {"max_completion_tokens": 800}
    kwargs.update(overrides)
    return client.call_model(MESSAGES, RESPONSE_FORMAT, **kwargs)


def sent_config(sdk) -> types.GenerateContentConfig:
    return sdk.models.generate_content.call_args.kwargs["config"]


# ------------------------------ request mapping ----------------------------- #
def test_messages_become_system_instruction_and_model_role_turns():
    system, contents = client._split_messages(MESSAGES)
    assert system == "SYS"
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1].parts[0].text == "draft"


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError):
        client._split_messages([{"role": "tool", "content": "x"}])


def test_request_carries_schema_json_mime_tokens_and_millisecond_timeout(fake):
    assert call(timeout=30.0) == '{"ok": true}'
    config = sent_config(fake)
    assert fake.models.generate_content.call_args.kwargs["model"] == client.DEFAULT_MODEL
    assert config.system_instruction == "SYS"
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == RESPONSE_FORMAT["json_schema"]["schema"]
    assert config.max_output_tokens == 800
    assert config.http_options.timeout == 30_000


def test_temperature_is_left_default_on_gemini_3_but_applied_on_older_models(fake, monkeypatch):
    call()
    assert sent_config(fake).temperature is None  # default model is gemini-3.x
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    call(temperature=0.3)
    assert sent_config(fake).temperature == 0.3
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GEMINI_TEMPERATURE", "0.5")
    call()
    assert sent_config(fake).temperature == 0.5  # explicit override wins everywhere


def test_thinking_level_adds_token_headroom_except_minimal(fake, monkeypatch):
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "minimal")
    call()
    assert sent_config(fake).max_output_tokens == 800
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "low")
    call()
    config = sent_config(fake)
    assert config.max_output_tokens == 800 + client.THINKING_HEADROOM_TOKENS
    assert config.thinking_config.thinking_level == types.ThinkingLevel.LOW


def test_invalid_thinking_level_is_reported(fake, monkeypatch):
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "extreme")
    with pytest.raises(client.LLMError, match="GEMINI_THINKING_LEVEL"):
        call()


def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    client._client.cache_clear()
    with pytest.raises(client.LLMError, match="GEMINI_API_KEY"):
        client._client()


# ------------------------------ response handling --------------------------- #
def test_blocked_prompt_raises(fake):
    fake.models.generate_content.return_value = types.GenerateContentResponse(
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason=types.BlockedReason.SAFETY)
    )
    with pytest.raises(client.LLMError, match="blocked the prompt.*SAFETY"):
        call()


def test_withheld_or_empty_response_raises(fake):
    fake.models.generate_content.return_value = gemini_response(None, finish=types.FinishReason.SAFETY)
    with pytest.raises(client.LLMError, match="SAFETY"):
        call()
    fake.models.generate_content.return_value = gemini_response("partial", finish=types.FinishReason.RECITATION)
    with pytest.raises(client.LLMError, match="withheld"):
        call()


def test_max_tokens_returns_partial_text_so_callers_can_retry(fake):
    fake.models.generate_content.return_value = gemini_response('{"title": "cut', finish=types.FinishReason.MAX_TOKENS)
    assert call() == '{"title": "cut'
    assert client.parse_json_object(call())[0] is None


# ------------------------------ errors / retries ---------------------------- #
def test_short_quota_wait_is_slept_then_retried(fake):
    fake.models.generate_content.side_effect = [quota_error(delay="3s"), gemini_response()]
    assert call() == '{"ok": true}'
    fake.sleep.assert_called_once()
    assert 3 <= fake.sleep.call_args.args[0] <= 4


def test_long_quota_wait_raises_rate_limited_without_sleeping(fake):
    fake.models.generate_content.side_effect = [quota_error(delay="60s")]
    with pytest.raises(client.LLMRateLimited) as info:
        call()
    assert info.value.retry_after == 60 and not info.value.daily
    fake.sleep.assert_not_called()


def test_daily_quota_is_not_retried(fake):
    fake.models.generate_content.side_effect = [
        quota_error(delay="2s", quota_id="GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    ]
    with pytest.raises(client.LLMRateLimited) as info:
        call()
    assert info.value.daily
    assert fake.models.generate_content.call_count == 1


def test_server_error_is_retried_then_succeeds(fake):
    unavailable = errors.ServerError(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}})
    fake.models.generate_content.side_effect = [unavailable, gemini_response()]
    assert call() == '{"ok": true}'
    assert fake.models.generate_content.call_count == 2


def test_client_error_is_not_retried(fake):
    bad = errors.ClientError(400, {"error": {"code": 400, "message": "bad schema", "status": "INVALID_ARGUMENT"}})
    fake.models.generate_content.side_effect = [bad]
    with pytest.raises(client.LLMError, match="400 INVALID_ARGUMENT: bad schema"):
        call()
    assert fake.models.generate_content.call_count == 1


def test_network_errors_are_retried_but_programming_errors_are_not(fake):
    fake.models.generate_content.side_effect = [TimeoutError("slow"), gemini_response()]
    assert call() == '{"ok": true}'
    fake.models.generate_content.side_effect = [KeyError("bug")]
    with pytest.raises(KeyError):
        call()


# ------------------------------ through the summarizer ---------------------- #
def test_summarizer_runs_end_to_end_on_the_gemini_client(fake):
    payload = json.dumps({
        "title": "مدل جدید",
        "hook": "یک جمله.",
        "bullets": ["الف", "ب", "ج"],
        "hashtags": ["هوش_مصنوعی", "AI", "LLM"],
    }, ensure_ascii=False)
    fake.models.generate_content.return_value = gemini_response(payload)
    item = NewsItem("hackernews:1", "hackernews", "Title", "https://example.com", "body")
    summary = summarizer.summarize(item)
    assert summary.bullets == ("الف", "ب", "ج")
    assert sent_config(fake).system_instruction == SYSTEM_PROMPT
    assert sent_config(fake).max_output_tokens == 800


# ------------------------------ configuration errors ------------------------- #
@pytest.mark.parametrize(
    "code, status, message",
    [
        (404, "NOT_FOUND", "models/gemini-1.5-flash is not found for API version v1beta"),  # a retired model name
        (403, "PERMISSION_DENIED", "The caller does not have permission"),
        (401, "UNAUTHENTICATED", "Request had invalid authentication credentials"),
        (400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key."),
    ],
)
def test_bad_key_or_model_is_a_config_error_and_is_never_retried(fake, code, status, message):
    fake.models.generate_content.side_effect = [errors.ClientError(code, {"error": {"code": code, "message": message, "status": status}})]
    with pytest.raises(client.LLMConfigError, match="GEMINI_API_KEY and GEMINI_MODEL"):
        call()
    assert fake.models.generate_content.call_count == 1


def test_ordinary_bad_requests_stay_plain_llm_errors(fake):
    bad = errors.ClientError(400, {"error": {"code": 400, "message": "schema too deep", "status": "INVALID_ARGUMENT"}})
    fake.models.generate_content.side_effect = [bad]
    with pytest.raises(client.LLMError) as info:
        call()
    assert not isinstance(info.value, client.LLMConfigError)


def test_a_missing_key_is_a_config_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    client._client.cache_clear()
    with pytest.raises(client.LLMConfigError):
        client._client()


# ------------------------------ outages -------------------------------------- #
def test_persistent_server_errors_become_llm_unavailable(fake):
    unavailable = errors.ServerError(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}})
    fake.models.generate_content.side_effect = [unavailable] * client.MAX_CALL_ATTEMPTS
    with pytest.raises(client.LLMUnavailable):
        call()
    assert fake.models.generate_content.call_count == client.MAX_CALL_ATTEMPTS


def test_persistent_network_failures_become_llm_unavailable(fake):
    fake.models.generate_content.side_effect = [TimeoutError("slow")] * client.MAX_CALL_ATTEMPTS
    with pytest.raises(client.LLMUnavailable):
        call()


def test_transient_errors_share_one_base_class_so_callers_can_treat_them_alike():
    assert issubclass(client.LLMRateLimited, client.LLMTransientError)
    assert issubclass(client.LLMUnavailable, client.LLMTransientError)
    assert not issubclass(client.LLMConfigError, client.LLMTransientError)
