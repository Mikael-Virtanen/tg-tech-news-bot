"""Gemini REST client tests. No network or API key needed: HTTP is replaced by a fake."""
import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.llm import client, summarizer
from src.llm.prompts import RESPONSE_FORMAT, SYSTEM_PROMPT
from src.models import NewsItem

MESSAGES = [
    {"role": "system", "content": "SYS"},
    {"role": "user", "content": "first"},
    {"role": "assistant", "content": "draft"},
    {"role": "user", "content": "fix it"},
]


def gemini_body(text: str | None = '{"ok": true}', finish: str = "STOP") -> dict:
    parts = [{"text": text}] if text is not None else []
    return {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": finish}]}


def rest_response(status=200, body=None) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.headers["Content-Type"] = "application/json"
    response._content = json.dumps(body if body is not None else gemini_body()).encode("utf-8")
    return response


def error_body(code, status, message, **extra):
    return {"error": {"code": code, "status": status, "message": message, **extra}}


def quota_error(message="Quota exceeded.", delay="1s", quota_id="GenerateRequestsPerMinutePerProjectPerModel-FreeTier"):
    return error_body(
        429, "RESOURCE_EXHAUSTED", message,
        details=[
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay},
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": quota_id}]},
        ],
    )


@pytest.fixture
def fake(monkeypatch):
    """Patch HTTP and clear env so each test controls its own configuration."""
    for var in ("GEMINI_MODEL", "GEMINI_THINKING_LEVEL", "GEMINI_TEMPERATURE", "GEMINI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    post = MagicMock(return_value=rest_response())
    session = MagicMock()
    session.post = post
    with patch.object(client, "_session", return_value=session), patch.object(client.time, "sleep") as sleep:
        post.sleep = sleep
        post.session = session
        yield post


def call(**overrides):
    kwargs = {"max_completion_tokens": 800}
    kwargs.update(overrides)
    return client.call_model(MESSAGES, RESPONSE_FORMAT, **kwargs)


def sent_payload(fake) -> dict:
    return fake.call_args.kwargs["json"]


def sent_headers(fake) -> dict:
    return fake.call_args.kwargs["headers"]


def sent_url(fake) -> str:
    return fake.call_args.args[0]


# ------------------------------ endpoint / headers --------------------------- #
def test_default_endpoint_is_googles_v1beta_and_carries_the_api_key_header(fake, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert call() == '{"ok": true}'
    assert sent_url(fake) == f"https://generativelanguage.googleapis.com/v1beta/models/{client.DEFAULT_MODEL}:generateContent"
    assert sent_headers(fake)["x-goog-api-key"] == "k"
    assert "Authorization" not in sent_headers(fake)    # Google endpoint: no Bearer header


def test_custom_base_url_is_used_verbatim_with_no_sdk_paths_added(fake, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://sinoxapi.com/v1/")  # trailing slash is noise
    call()
    assert sent_url(fake) == f"https://sinoxapi.com/v1/models/{client.DEFAULT_MODEL}:generateContent"
    assert sent_headers(fake)["x-goog-api-key"] == "k"
    assert sent_headers(fake)["Authorization"] == "Bearer k"  # gateways read this one

    monkeypatch.setenv("GEMINI_BASE_URL", "")  # GitHub passes unset variables as ""
    call()
    assert sent_url(fake).startswith("https://generativelanguage.googleapis.com/v1beta/")


# ------------------------------ request mapping ----------------------------- #


# ------------------------------ request mapping ----------------------------- #
def test_messages_become_system_instruction_and_model_role_turns():
    system, contents = client._split_messages(MESSAGES)
    assert system == "SYS"
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert contents[1]["parts"][0]["text"] == "draft"


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError):
        client._split_messages([{"role": "tool", "content": "x"}])


def test_request_carries_schema_config_and_system_instruction(fake):
    call(timeout=30.0)
    payload = sent_payload(fake)
    assert payload["systemInstruction"]["parts"][0]["text"] == "SYS"
    config = payload["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"] == RESPONSE_FORMAT["json_schema"]["schema"]
    assert config["maxOutputTokens"] == 800


def test_temperature_is_left_default_on_gemini_3_but_applied_on_older_models(fake, monkeypatch):
    call()
    assert "temperature" not in sent_payload(fake)["generationConfig"]  # default model is gemini-3.x
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    call(temperature=0.3)
    assert sent_payload(fake)["generationConfig"]["temperature"] == 0.3
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GEMINI_TEMPERATURE", "0.5")
    call()
    assert sent_payload(fake)["generationConfig"]["temperature"] == 0.5  # explicit override wins everywhere


def test_thinking_level_adds_token_headroom_except_minimal(fake, monkeypatch):
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "minimal")
    call()
    assert sent_payload(fake)["generationConfig"]["maxOutputTokens"] == 800
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "low")
    call()
    config = sent_payload(fake)["generationConfig"]
    assert config["maxOutputTokens"] == 800 + client.THINKING_HEADROOM_TOKENS
    assert config["thinkingConfig"]["thinkingLevel"] == "low"


def test_invalid_thinking_level_is_reported(fake, monkeypatch):
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "extreme")
    with pytest.raises(client.LLMError, match="GEMINI_THINKING_LEVEL"):
        call()


def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(client.LLMError, match="GEMINI_API_KEY"):
        client._api_key()


# ------------------------------ response handling --------------------------- #
def test_snake_case_gateway_responses_are_read_tolerantly(fake):
    fake.return_value = rest_response(body={
        "candidates": [{"finish_reason": "STOP", "content": {"role": "model", "parts": [{"text": "snake"}]}}],
    })
    assert call() == "snake"


def test_blocked_prompt_raises(fake):
    fake.return_value = rest_response(body={"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(client.LLMError, match="blocked the prompt.*SAFETY"):
        call()


def test_withheld_or_empty_response_raises(fake):
    fake.return_value = rest_response(body=gemini_body(None, finish="SAFETY"))
    with pytest.raises(client.LLMError, match="SAFETY"):
        call()
    fake.return_value = rest_response(body=gemini_body("partial", finish="RECITATION"))
    with pytest.raises(client.LLMError, match="withheld"):
        call()


def test_max_tokens_returns_partial_text_so_callers_can_retry(fake):
    fake.return_value = rest_response(body=gemini_body('{"title": "cut', finish="MAX_TOKENS"))
    assert call() == '{"title": "cut'
    assert client.parse_json_object(call())[0] is None


def test_non_json_success_body_is_a_plain_llm_error(fake):
    response = requests.Response()
    response.status_code = 200
    response._content = b"<html>gateway oops</html>"
    fake.return_value = response
    with pytest.raises(client.LLMError, match="non-JSON"):
        call()


# ------------------------------ errors / retries ---------------------------- #
def test_short_quota_wait_is_slept_then_retried(fake):
    fake.side_effect = [rest_response(429, quota_error(delay="3s")), rest_response()]
    assert call() == '{"ok": true}'
    fake.sleep.assert_called_once()
    assert 3 <= fake.sleep.call_args.args[0] <= 4


def test_long_quota_wait_raises_rate_limited_without_sleeping(fake):
    fake.return_value = rest_response(429, quota_error(delay="60s"))
    with pytest.raises(client.LLMRateLimited) as info:
        call()
    assert info.value.retry_after == 60 and not info.value.daily
    fake.sleep.assert_not_called()


def test_daily_quota_is_not_retried(fake):
    fake.return_value = rest_response(429, quota_error(delay="2s", quota_id="GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
    with pytest.raises(client.LLMRateLimited) as info:
        call()
    assert info.value.daily
    assert fake.call_count == 1


def test_gateway_429_without_a_json_body_is_still_rate_limiting(fake):
    response = requests.Response()
    response.status_code = 429
    response._content = b"rate limited"
    fake.return_value = response
    with pytest.raises(client.LLMRateLimited):
        call()


def test_server_error_is_retried_then_succeeds(fake):
    fake.side_effect = [rest_response(503, error_body(503, "UNAVAILABLE", "overloaded")), rest_response()]
    assert call() == '{"ok": true}'
    assert fake.call_count == 2


def test_client_error_is_not_retried(fake):
    fake.return_value = rest_response(400, error_body(400, "INVALID_ARGUMENT", "bad schema"))
    with pytest.raises(client.LLMError, match="400 INVALID_ARGUMENT: bad schema"):
        call()
    assert fake.call_count == 1


def test_network_errors_are_retried(fake):
    fake.side_effect = [requests.ConnectionError("reset"), rest_response()]
    assert call() == '{"ok": true}'
    fake.side_effect = [requests.Timeout("slow"), rest_response()]
    assert call() == '{"ok": true}'


def test_persistent_network_failures_become_llm_unavailable(fake):
    fake.side_effect = [requests.Timeout("slow")] * client.MAX_CALL_ATTEMPTS
    with pytest.raises(client.LLMUnavailable):
        call()


def test_404_is_treated_as_a_config_error_never_retried(fake):
    fake.return_value = rest_response(404, error_body(404, "NOT_FOUND", "no such model"))
    with pytest.raises(client.LLMConfigError, match="GEMINI_API_KEY"):
        call()
    assert fake.call_count == 1


def test_a_404_from_a_gateway_that_does_not_serve_the_path_names_the_base_url(fake, monkeypatch):
    monkeypatch.setenv("GEMINI_BASE_URL", "https://sinoxapi.com/v1")
    fake.return_value = rest_response(404, error_body(404, "NOT_FOUND", "no such model"))
    with pytest.raises(client.LLMConfigError, match="GEMINI_BASE_URL"):
        call()
    assert fake.call_count == 1


# ------------------------------ error-body dialects -------------------------- #
@pytest.mark.parametrize("body", [
    {"error": "Invalid API key provided"},          # the crash: a plain string, not a dict
    {"error": ""},                                  # empty string
    {"error": 42},                                  # a number
    {"error": ["a", "b"]},                          # a list
    {"error": None},                                # explicit null
    {"message": "model not found"},                 # top-level message, OpenAI style
    [1, 2, 3],                                      # a JSON array body
    "null",                                          # JSON null
])
def test_every_error_body_shape_is_handled_without_crashing(fake, body):
    fake.return_value = rest_response(400, body)
    try:
        call()
        assert False, "a 400 must raise"
    except client.LLMError as exc:
        assert "400" in str(exc)  # classified, with the HTTP status as the code
    assert fake.call_count == 1


def test_string_code_inside_the_error_dict_is_coerced(fake):
    fake.return_value = rest_response(200, error_body("429", "RESOURCE_EXHAUSTED", "quota by string code"))
    with pytest.raises(client.LLMRateLimited):
        call()


# ------------------------------ through the summarizer ---------------------- #
def test_summarizer_runs_end_to_end_on_the_rest_client(fake):
    payload = json.dumps({
        "title": "مدل جدید",
        "hook": "یک جمله.",
        "bullets": ["الف", "ب", "ج"],
        "hashtags": ["هوش_مصنوعی", "AI", "LLM"],
    }, ensure_ascii=False)
    fake.return_value = rest_response(body=gemini_body(payload))
    item = NewsItem("hackernews:1", "hackernews", "Title", "https://example.com", "body")
    summary = summarizer.summarize(item)
    assert summary.bullets == ("الف", "ب", "ج")
    assert sent_payload(fake)["systemInstruction"]["parts"][0]["text"] == SYSTEM_PROMPT
    assert sent_payload(fake)["generationConfig"]["maxOutputTokens"] == 800


# ------------------------------ configuration errors ------------------------- #
@pytest.mark.parametrize("code, status, message", [
    (403, "PERMISSION_DENIED", "The caller does not have permission"),
    (401, "UNAUTHENTICATED", "Request had invalid authentication credentials"),
    (400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key."),
])
def test_bad_key_or_model_is_a_config_error_and_is_never_retried(fake, code, status, message):
    fake.return_value = rest_response(code, error_body(code, status, message))
    with pytest.raises(client.LLMConfigError, match="GEMINI_API_KEY and GEMINI_MODEL"):
        call()
    assert fake.call_count == 1


@pytest.mark.parametrize("body, code, expected_in_message", [
    ({"error": "rate limit exceeded, retry later"}, 429, "rate limit exceeded"),  # error is a plain STRING
    ({"error": None}, 404, "HTTP 404"),                                           # error is null
    ({"message": "upstream unavailable"}, 503, "upstream unavailable"),           # no "error" key at all
    ({"error": {"code": "429", "message": "weird string code"}}, 429, "weird string code"),  # code as a string
    ({"error": {"code": "not-a-number", "message": "boom"}}, 500, "boom"),        # unparseable code
])
def test_gateway_error_shapes_never_crash_the_handler(fake, body, code, expected_in_message):
    """The original bug: a proxy returned {"error": "text"} and _error_fields called .get on a str."""
    fake.return_value = rest_response(code, body)
    try:
        call()
    except client.LLMError as exc:  # some of these raise LLMRateLimited / LLMConfigError / LLMUnavailable subclasses
        assert expected_in_message in str(exc)
    else:
        pytest.fail("a non-2xx response must always raise")


def test_plain_text_error_body_never_crashes_the_handler(fake):
    response = requests.Response()
    response.status_code = 502
    response._content = b"Bad Gateway"
    fake.return_value = response
    with pytest.raises(client.LLMUnavailable, match="Bad Gateway"):
        call()


def test_weird_success_bodies_never_crash_extraction(fake):
    """candidates/content in shapes the schema never promised: dict, string, null, or a bare part."""
    fake.return_value = rest_response(body={"candidates": {}})
    with pytest.raises(client.LLMError, match="no candidates"):
        call()
    fake.return_value = rest_response(body={"candidates": ["garbage"]})
    with pytest.raises(client.LLMError, match="no text"):
        call()
    fake.return_value = rest_response(body={"candidates": [{"content": "not-a-dict", "finishReason": "STOP"}]})
    with pytest.raises(client.LLMError, match="no text"):
        call()
    fake.return_value = rest_response(body={"candidates": [{"content": None, "finishReason": "STOP"}]})
    with pytest.raises(client.LLMError, match="no text"):
        call()
    fake.return_value = rest_response(400, error_body(400, "INVALID_ARGUMENT", "schema too deep"))
    with pytest.raises(client.LLMError) as info:
        call()
    assert not isinstance(info.value, client.LLMConfigError)


def test_a_missing_key_is_a_config_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(client.LLMConfigError):
        client.call_model([{"role": "user", "content": "x"}], {"json_schema": {"schema": {}}}, max_completion_tokens=10)


# ------------------------------ outages -------------------------------------- #
def test_persistent_server_errors_become_llm_unavailable(fake):
    fake.return_value = rest_response(503, error_body(503, "UNAVAILABLE", "overloaded"))
    with pytest.raises(client.LLMUnavailable):
        call()
    assert fake.call_count == client.MAX_CALL_ATTEMPTS


def test_transient_errors_share_one_base_class_so_callers_can_treat_them_alike():
    assert issubclass(client.LLMRateLimited, client.LLMTransientError)
    assert issubclass(client.LLMUnavailable, client.LLMTransientError)
    assert not issubclass(client.LLMConfigError, client.LLMTransientError)
