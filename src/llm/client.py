"""
LLM transport built on plain `requests`, with two backends chosen automatically:

    GEMINI_BASE_URL set     -> OpenAI-compatible chat/completions (OneAPI / Sinox-style gateways)
    GEMINI_BASE_URL unset   -> Google's native Gemini REST API (the free-tier default)

Why not the official `google-genai` SDK? It owns URL construction, session handling and
auth headers, so against a proxy that re-publishes models at a custom base URL it calls
paths the proxy does not serve (404s, rejected keys). OpenAI-compatible gateways (OneAPI
and friends) are even further away: they want `{base_url}/chat/completions` with OpenAI's
payload. Plain HTTP gives exact control over the request shape, whichever backend is used:

    OpenAI (custom base):   POST {base_url}/chat/completions
                            {"model": ..., "messages": [{"role","content"}, ...],
                             "response_format": {"type": "json_object"},
                             "max_tokens": ...}
                            Authorization: Bearer <key>
    Gemini (default):       POST {base}/v1beta/models/{model}:generateContent
                            {"contents": [...], "systemInstruction": ..., "generationConfig": ...}
                            x-goog-api-key: <key>

The rest of the LLM layer is backend-agnostic: it already passes OpenAI-style `messages`
(system / user / assistant) and the OpenAI-style `response_format` dict from llm/prompts.py.
For the OpenAI backend they pass through almost verbatim (the JSON schema is kept in the
prompt, and the response is asked to be a plain JSON object — "strict json_schema" is a
chat-completions extension many gateways don't implement). For Gemini they are translated.

JSON-mode note for the OpenAI backend: `json_object` requires the word "JSON" to appear
in the messages (OpenAI's contract) — the prompts contain it. The schema is additionally
inlined into the system prompt by build_user_message callers, so gateways without
json_object support still see it.

Environment:
    GEMINI_API_KEY          key for Google (https://aistudio.google.com/apikey, GOOGLE_API_KEY
                            also works) or the key issued by the proxy/gateway
    GEMINI_BASE_URL         base URL of an OpenAI-compatible proxy, e.g. https://sinoxapi.com/v1 ;
                            empty string counts as unset (GitHub passes unset variables as "")
    GEMINI_MODEL            optional, defaults to DEFAULT_MODEL (set it to the proxy's name,
                            e.g. t-gemini-3.8-flash, when using a gateway)
    GEMINI_THINKING_LEVEL   optional: minimal | low | medium | high (Gemini 3.x models; native
                            backend only — sent as extra body params on the OpenAI backend
                            only if GEMINI_OPENAI_EXTRA is set)
    GEMINI_TEMPERATURE      optional float; overrides the callers' temperature on every model

Free-tier notes (native Google backend):
    - Per-model RPM/RPD limits are not published in the docs; they are shown for your
      project at https://aistudio.google.com/rate-limit. Quota errors (HTTP 429) surface
      as `LLMRateLimited`; daily and per-minute quotas are distinguished because a daily
      quota must never be hammered.
    - On the free tier Google may use prompts and outputs to improve its products. Only send
      public content (news, abstracts, READMEs), never private data.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from functools import lru_cache
from typing import Any

import requests

# GA, free tier, thinks "minimal" by default, so the output-token cap isn't eaten by reasoning.
# gemini-1.5-* and gemini-2.0-* have been shut down; verify the current list at
# https://ai.google.dev/gemini-api/docs/deprecations before changing this.
# With an OpenAI-compatible gateway set GEMINI_MODEL to the name the gateway knows (t-...).
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

MAX_CALL_ATTEMPTS = 3          # 1 call + 2 retries for transient failures
MAX_RETRY_WAIT_SECONDS = 20.0  # longer quota waits are raised, not slept (CI minutes)
THINKING_HEADROOM_TOKENS = 2048  # added to maxOutputTokens when thinking is enabled (native backend)
CONNECT_TIMEOUT = 10.0
THINKING_LEVELS = ("minimal", "low", "medium", "high")

# OpenAI finish reasons (and Gemini's) meaning "the content was withheld", not "the model finished".
_WITHHELD_FINISH_REASONS = {
    "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "LANGUAGE", "OTHER",
    "content_filter", "sensitive",
}

# Gateways rename models (t-gemini-3.8-flash, gemini-3.8-flash@free, provider/gemini-3...):
# "gemini-3" anywhere in the name counts, so Google's leave-temperature-at-default advice
# for Gemini 3.x still applies behind a prefix.
_GEMINI3_RE = re.compile(r"gemini-3")

logger = logging.getLogger(__name__)

Messages = list[dict[str, str]]


class LLMError(RuntimeError):
    """Base class for every LLM-layer failure. Handlers can catch this one type and skip the item."""


class LLMConfigError(LLMError):
    """Missing/invalid API key, unknown model name, ...: nothing about the item is wrong and retrying
    cannot help. The run must fail loudly; it must never be recorded against the item being processed."""


class LLMTransientError(LLMError):
    """The service, not the item, is the problem (quota, outage). The run stops cleanly and the item
    is left untouched so the next scheduled run simply tries again."""


class LLMUnavailable(LLMTransientError):
    """Network failure or HTTP 5xx that persisted through the retries."""


class LLMRateLimited(LLMTransientError):
    """Quota exhausted (per-minute or daily). The item is fine; leave it for the next run
    instead of marking it failed."""

    def __init__(self, message: str, *, retry_after: float | None = None, daily: bool = False) -> None:
        super().__init__(message)
        self.retry_after = retry_after  # seconds, if the server said how long
        self.daily = daily              # True if the per-day quota looks exhausted


# --------------------------------------------------------------------------- #
# HTTP plumbing
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    return session


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise LLMConfigError("GEMINI_API_KEY is not set (create a free key at https://aistudio.google.com/apikey)")
    return key


def _base_url() -> str:
    """Custom endpoint (proxy/gateway) or Google's default. Empty string counts as unset."""
    return (os.environ.get("GEMINI_BASE_URL") or "").strip().rstrip("/") or DEFAULT_BASE_URL


def _custom_endpoint() -> bool:
    return bool((os.environ.get("GEMINI_BASE_URL") or "").strip())


def _endpoint(model: str) -> str:
    """The exact request URL: {base}/chat/completions for gateways, the Gemini path for Google."""
    if _custom_endpoint():
        return f"{_base_url()}/chat/completions"
    return f"{_base_url()}/models/{model}:generateContent"


def _headers(api_key: str) -> dict[str, str]:
    if _custom_endpoint():
        # OpenAI-compatible gateways read Authorization: Bearer; sending x-goog-api-key too
        # costs nothing and keeps a native-Gemini mirror of the gateway working.
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "x-goog-api-key": api_key,
        }
    return {"Content-Type": "application/json", "x-goog-api-key": api_key}


# --------------------------------------------------------------------------- #
# Request building
# --------------------------------------------------------------------------- #
def _validate_roles(messages: Messages) -> None:
    for message in messages:
        if message["role"] not in ("system", "user", "assistant"):
            raise ValueError(f"unsupported message role: {message['role']!r}")


def _split_messages(messages: Messages) -> tuple[str | None, list[dict[str, Any]]]:
    """OpenAI-style messages -> (system text, Gemini REST `contents`)."""
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        role, text = message["role"], message["content"]
        if role == "system":
            system_parts.append(text)
        else:
            contents.append({"role": "user" if role == "user" else "model", "parts": [{"text": text}]})
    return ("\n\n".join(system_parts) or None), contents


def _json_schema(response_format: dict[str, Any]) -> dict[str, Any]:
    try:
        return response_format["json_schema"]["schema"]
    except (KeyError, TypeError) as exc:
        raise ValueError("response_format must look like {'json_schema': {'schema': {...}}}") from exc


def _openai_payload(
    messages: Messages,
    *,
    model: str,
    max_completion_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """OpenAI chat/completions body. Messages pass through verbatim (they already are OpenAI-style).

    `response_format: json_object` asks for JSON output; the schema itself is part of the system
    prompt, so gateways without structured-output support still follow it.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": [dict(m) for m in messages],
        "response_format": {"type": "json_object"},
        "max_tokens": max_completion_tokens,
    }

    # Google recommends leaving temperature at its default (1.0) on Gemini 3.x models (matched
    # anywhere in the name, so renamed gateway models still qualify); the caller's temperature is
    # only sent otherwise, unless GEMINI_TEMPERATURE overrides everything.
    override = os.environ.get("GEMINI_TEMPERATURE")
    if override:
        payload["temperature"] = float(override)
    elif not _GEMINI3_RE.search(model):
        payload["temperature"] = temperature
    return payload


def _gemini_payload(
    messages: Messages,
    response_format: dict[str, Any],
    *,
    model: str,
    max_completion_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    system, contents = _split_messages(messages)
    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": _generation_config(
            response_format, model=model, max_completion_tokens=max_completion_tokens, temperature=temperature
        ),
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    return payload


def _generation_config(
    response_format: dict[str, Any],
    *,
    model: str,
    max_completion_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "responseMimeType": "application/json",
        "responseJsonSchema": _json_schema(response_format),
        "maxOutputTokens": max_completion_tokens,
    }

    override = os.environ.get("GEMINI_TEMPERATURE")
    if override:
        config["temperature"] = float(override)
    elif not _GEMINI3_RE.search(model):
        config["temperature"] = temperature

    level = (os.environ.get("GEMINI_THINKING_LEVEL") or "").strip().lower()
    if level:
        if level not in THINKING_LEVELS:
            raise LLMConfigError(f"invalid GEMINI_THINKING_LEVEL {level!r} (use one of {THINKING_LEVELS})")
        config["thinkingConfig"] = {"thinkingLevel": level}
        if level != "minimal":
            config["maxOutputTokens"] = max_completion_tokens + THINKING_HEADROOM_TOKENS
    return config


# --------------------------------------------------------------------------- #
# Error classification
# --------------------------------------------------------------------------- #
_RETRY_IN_RE = re.compile(r"retry in ([\d.]+)\s*s", re.IGNORECASE)
_RETRY_DELAY_RE = re.compile(r"""retryDelay['"]?\s*:\s*['"]?([\d.]+)\s*s""")


def _retry_delay_seconds(details: Any, message: str) -> float | None:
    """Extract the server's suggested wait from a 429 (RetryInfo detail, OpenAI style, or free text)."""
    blob = f"{json.dumps(details, default=str)} {message}"
    for pattern in (_RETRY_DELAY_RE, _RETRY_IN_RE):
        match = pattern.search(blob)
        if match:
            return float(match.group(1))
    return None


def _is_daily_quota(blob: str) -> bool:
    # Heuristic: quota IDs for daily limits look like "GenerateRequestsPerDayPerProjectPerModel-FreeTier".
    return "PerDay" in blob


def _error_fields(response: requests.Response) -> tuple[int, str, str, list]:
    """(code, status, message, details) from an error body, tolerating every shape a gateway emits.

    Seen in the wild: {"error": {code, status, message, details}} (Google), {"error": "text"}
    (gateways), {"message": "..."} (OpenAI style), a non-JSON/HTML body, or an unparseable body.
    Fields are read defensively; anything unreadable falls back to the HTTP status itself.
    """
    try:
        data = response.json()
    except ValueError:
        data = None
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict):
        raw_code = error.get("code") or response.status_code
        try:
            code = int(raw_code)
        except (TypeError, ValueError):
            code = response.status_code
        status = str(error.get("status") or f"HTTP_{response.status_code}")
        message = str(error.get("message") or (response.text or "")[:500] or f"HTTP {response.status_code}")
        details = error.get("details") or []
        if not isinstance(details, list):
            details = []
    elif isinstance(error, str) and error.strip():
        # {"error": "something went wrong"} — a plain string instead of the Google object
        code = response.status_code
        status = f"HTTP_{response.status_code}"
        message = error.strip()[:500]
        details = []
    else:
        # a bare JSON body without an "error" key ({"message": "..."}), non-JSON, or unparseable
        code = response.status_code
        status = f"HTTP_{response.status_code}"
        top_message = data.get("message") if isinstance(data, dict) else None
        fallback = str(top_message or (response.text or "").strip()[:500] or f"HTTP {response.status_code}")
        message = fallback
        details = []
    return code, status, message, details


# --------------------------------------------------------------------------- #
# Response handling
# --------------------------------------------------------------------------- #
def _first(body: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in body:
            return body[name]
    return None


def _extract_text_gemini(body: dict[str, Any]) -> str:
    candidates = body.get("candidates") or []
    if not isinstance(candidates, list) or not candidates:
        feedback = _first(body, "promptFeedback", "prompt_feedback")
        reason = _first(feedback, "blockReason", "block_reason") if isinstance(feedback, dict) else None
        raise LLMError(f"Gemini blocked the prompt (blockReason={reason})" if reason else "Gemini returned no candidates")

    candidate = candidates[0] if isinstance(candidates[0], dict) else {}
    finish = str(_first(candidate, "finishReason", "finish_reason") or "")
    content = _first(candidate, "content")
    parts = content.get("parts") if isinstance(content, dict) else None
    text = "".join(str(p.get("text", "")) for p in (parts or []) if isinstance(p, dict))
    if not text:
        raise LLMError(f"Gemini returned no text (finishReason={finish})")
    if finish in _WITHHELD_FINISH_REASONS:
        raise LLMError(f"Gemini withheld the response (finishReason={finish})")
    # finish == MAX_TOKENS returns the partial text on purpose: it fails JSON parsing, which the
    # callers turn into a corrective retry.
    return text


def _extract_text_openai(body: dict[str, Any]) -> str:
    """choices[0].message.content, tolerating gateways that stray from the spec."""
    choices = body.get("choices") or []
    if not isinstance(choices, list) or not choices:
        error = body.get("error")
        if isinstance(error, str) and error.strip():
            raise LLMError(f"the gateway returned an error envelope: {error[:300]}")
        if isinstance(error, dict) and error.get("message"):
            raise LLMError(f"the gateway returned an error envelope: {str(error['message'])[:300]}")
        raise LLMError("the gateway returned no choices (unexpected response shape)")

    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    if not isinstance(message, dict):
        message = {}
    text = message.get("content")
    if isinstance(text, list):  # some multi-content dialects
        text = "".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in text)
    finish = str(choice.get("finish_reason") or "")
    if text is None:
        # An assistant message with tool calls or a refusal instead of content
        refusal = message.get("refusal")
        if refusal:
            raise LLMError(f"the model refused to answer: {str(refusal)[:300]}")
        raise LLMError(f"the gateway returned no message content (finish_reason={finish})")
    text = str(text)
    if not text:
        raise LLMError(f"the gateway returned empty message content (finish_reason={finish})")
    if finish in _WITHHELD_FINISH_REASONS:
        raise LLMError(f"the model withheld the response (finish_reason={finish})")
    # finish == length returns the partial text on purpose: it fails JSON parsing, which the
    # callers turn into a corrective retry.
    return text


def _log_usage(kind: str, body: dict[str, Any]) -> None:
    usage = _first(body, "usage", "usageMetadata", "usage_metadata")
    if isinstance(usage, dict):
        logger.debug(
            "%s usage: prompt=%s output=%s total=%s",
            kind,
            usage.get("prompt_tokens", usage.get("promptTokenCount", usage.get("prompt_token_count"))),
            usage.get("completion_tokens", usage.get("candidatesTokenCount", usage.get("candidates_token_count"))),
            usage.get("total_tokens", usage.get("totalTokenCount", usage.get("total_token_count"))),
        )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def call_model(
    messages: Messages,
    response_format: dict[str, Any],
    *,
    max_completion_tokens: int,
    timeout: float = 30.0,
    temperature: float = 0.3,
) -> str:
    """Single generation (with transient-error retries); returns the raw JSON text."""
    _validate_roles(messages)
    model = os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL  # `or`: GitHub passes unset variables as ""
    api_key = _api_key()
    custom = _custom_endpoint()

    kind = "gateway" if custom else "Gemini"
    if custom:
        payload = _openai_payload(messages, model=model, max_completion_tokens=max_completion_tokens, temperature=temperature)
    else:
        payload = _gemini_payload(messages, response_format, model=model, max_completion_tokens=max_completion_tokens, temperature=temperature)

    endpoint = _endpoint(model)
    headers = _headers(api_key)
    extract = _extract_text_openai if custom else _extract_text_gemini

    for attempt in range(1, MAX_CALL_ATTEMPTS + 1):
        last_attempt = attempt == MAX_CALL_ATTEMPTS
        try:
            response = _session().post(endpoint, json=payload, headers=headers, timeout=(CONNECT_TIMEOUT, timeout))
        except requests.RequestException as exc:
            if last_attempt:
                raise LLMUnavailable(f"{kind} request failed: {exc}") from exc
            logger.warning("%s network error (%s); retrying (attempt %d)", kind, exc, attempt)
            time.sleep(2 ** attempt)
            continue

        if response.status_code == 200:
            try:
                body = response.json()
            except ValueError as exc:
                raise LLMError(f"{kind} returned a non-JSON body: {(response.text or '')[:200]!r}") from exc
            if not isinstance(body, dict):
                raise LLMError(f"{kind} returned an unexpected JSON body (not an object)")
            _log_usage(kind, body)
            return extract(body)

        code, status, message, details = _error_fields(response)
        if code == 429:
            delay = _retry_delay_seconds(details, message)
            daily = _is_daily_quota(f"{message} {json.dumps(details, default=str)}")
            if not daily and not last_attempt and delay is not None and delay <= MAX_RETRY_WAIT_SECONDS:
                logger.warning("%s quota hit; retrying in %.1fs (attempt %d)", kind, delay, attempt)
                time.sleep(delay + 0.5)
                continue
            raise LLMRateLimited(
                f"{kind} quota exhausted ({'daily' if daily else 'per-minute'}): {message}",
                retry_after=delay, daily=daily,
            )
        if code in (500, 502, 503, 504) and not last_attempt:
            logger.warning("%s %s; retrying (attempt %d)", kind, code, attempt)
            time.sleep(2 ** attempt)
            continue

        detail = f"{kind} API error {code} {status}: {message}"
        if code in (500, 502, 503, 504):  # the retries above are exhausted
            raise LLMUnavailable(detail)
        if code in (401, 403, 404) or (code == 400 and "api key" in message.lower()):
            # bad/unauthorized key, or a model/endpoint that doesn't exist (404)
            raise LLMConfigError(
                f"{detail} (check GEMINI_API_KEY and GEMINI_MODEL"
                + (", and that GEMINI_BASE_URL points at an OpenAI-compatible gateway" if custom else "")
                + ")"
            )
        raise LLMError(detail)

    raise LLMError(f"{kind} request failed")  # pragma: no cover - loop always returns or raises


def parse_json_object(raw: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Return (data, []) on success or (None, [problem]) in a form that can be fed back to the model."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, ["the response was not valid JSON (it may have been cut off); return the complete JSON object"]
    if not isinstance(data, dict):
        return None, ["the response must be a single JSON object"]
    return data, []
