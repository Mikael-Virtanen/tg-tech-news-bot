"""
Gemini REST client built on plain `requests`.

Why not the official `google-genai` SDK? It owns URL construction, session handling and
auth headers, so against a proxy that re-publishes the Gemini API at a custom base URL it
calls paths the proxy does not serve (hence 404s and rejected keys). Plain HTTP gives
exact control over the ONE request shape this bot needs:

    POST {base_url}/models/{model}:generateContent
    headers: x-goog-api-key (always; Google's header)
             Authorization: Bearer (custom base URLs only: OpenAI-style gateways read this one)

    GEMINI_BASE_URL=https://sinoxapi.com/v1 -> https://sinoxapi.com/v1/models/<model>:generateContent
    GEMINI_BASE_URL unset or empty          -> https://generativelanguage.googleapis.com/v1beta/models/...

The rest of the LLM layer stays backend-agnostic: it passes OpenAI-style `messages`
(system / user / assistant) and the OpenAI-style `response_format` dict from
llm/prompts.py. This module is the ONLY place that knows about Gemini, and translates:

    system messages        -> systemInstruction.parts
    user / assistant turns -> contents with roles "user" / "model"
    response_format schema -> generationConfig.responseJsonSchema

CamelCase JSON names are used (the canonical protobuf-JSON mapping); response fields are
read tolerantly, accepting snake_case too, because gateways emit both.

Environment:
    GEMINI_API_KEY          key for Google (https://aistudio.google.com/apikey, GOOGLE_API_KEY
                            also works) or the key issued by the proxy
    GEMINI_BASE_URL         optional proxy/gateway base URL, e.g. https://sinoxapi.com/v1 ;
                            empty string counts as unset (GitHub passes unset variables as "")
    GEMINI_MODEL            optional, defaults to DEFAULT_MODEL
    GEMINI_THINKING_LEVEL   optional: minimal | low | medium | high (Gemini 3.x models)
    GEMINI_TEMPERATURE      optional float; overrides the callers' temperature on every model

Free-tier notes:
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
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

MAX_CALL_ATTEMPTS = 3          # 1 call + 2 retries for transient failures
MAX_RETRY_WAIT_SECONDS = 20.0  # longer quota waits are raised, not slept (CI minutes)
THINKING_HEADROOM_TOKENS = 2048  # added to maxOutputTokens when thinking is enabled
CONNECT_TIMEOUT = 10.0
THINKING_LEVELS = ("minimal", "low", "medium", "high")

# Finish reasons meaning "the content was withheld", not "the model finished".
_WITHHELD_FINISH_REASONS = {
    "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "LANGUAGE", "OTHER",
}

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
    """Gemini quota exhausted. The item is fine; leave it for the next run instead of marking it failed."""

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
    """The exact request URL; the SDK's path assembly is exactly what proxies choke on."""
    return f"{_base_url()}/models/{model}:generateContent"


def _headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    if _custom_endpoint():
        # Google reads x-goog-api-key; OpenAI-style gateways read Authorization. Sending both
        # costs nothing and lets one GEMINI_API_KEY work against either.
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


# --------------------------------------------------------------------------- #
# Request building
# --------------------------------------------------------------------------- #
def _split_messages(messages: Messages) -> tuple[str | None, list[dict[str, Any]]]:
    """OpenAI-style messages -> (system text, REST `contents`)."""
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        role, text = message["role"], message["content"]
        if role == "system":
            system_parts.append(text)
        elif role in ("user", "assistant"):
            contents.append({"role": "user" if role == "user" else "model", "parts": [{"text": text}]})
        else:
            raise ValueError(f"unsupported message role: {role!r}")
    return ("\n\n".join(system_parts) or None), contents


def _json_schema(response_format: dict[str, Any]) -> dict[str, Any]:
    try:
        return response_format["json_schema"]["schema"]
    except (KeyError, TypeError) as exc:
        raise ValueError("response_format must look like {'json_schema': {'schema': {...}}}") from exc


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

    # Google recommends leaving temperature at its default (1.0) on Gemini 3.x models, so the
    # callers' value is only applied to older models unless GEMINI_TEMPERATURE says otherwise.
    override = os.environ.get("GEMINI_TEMPERATURE")
    if override:
        config["temperature"] = float(override)
    elif not model.startswith("gemini-3"):
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
    """Extract the server's suggested wait from a 429 (RetryInfo detail or the message text)."""
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
    (gateways), a bare "text" body, a non-JSON body, or an all-200-but-error envelope. Fields
    are read defensively; anything unreadable falls back to the HTTP status itself.
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


def _extract_text(body: dict[str, Any]) -> str:
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


def _log_usage(body: dict[str, Any]) -> None:
    usage = _first(body, "usageMetadata", "usage_metadata")
    if isinstance(usage, dict):
        logger.debug(
            "Gemini usage: prompt=%s output=%s total=%s",
            usage.get("promptTokenCount", usage.get("prompt_token_count")),
            usage.get("candidatesTokenCount", usage.get("candidates_token_count")),
            usage.get("totalTokenCount", usage.get("total_token_count")),
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
    model = os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL  # `or`: GitHub passes unset variables as ""
    api_key = _api_key()
    system, contents = _split_messages(messages)

    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": _generation_config(
            response_format, model=model, max_completion_tokens=max_completion_tokens, temperature=temperature
        ),
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    endpoint = _endpoint(model)
    headers = _headers(api_key)

    for attempt in range(1, MAX_CALL_ATTEMPTS + 1):
        last_attempt = attempt == MAX_CALL_ATTEMPTS
        try:
            response = _session().post(endpoint, json=payload, headers=headers, timeout=(CONNECT_TIMEOUT, timeout))
        except requests.RequestException as exc:
            if last_attempt:
                raise LLMUnavailable(f"Gemini request failed: {exc}") from exc
            logger.warning("Gemini network error (%s); retrying (attempt %d)", exc, attempt)
            time.sleep(2 ** attempt)
            continue

        if response.status_code == 200:
            try:
                body = response.json()
            except ValueError as exc:
                raise LLMError(f"Gemini returned a non-JSON body: {(response.text or '')[:200]!r}") from exc
            if not isinstance(body, dict):
                raise LLMError("Gemini returned an unexpected JSON body (not an object)")
            _log_usage(body)
            return _extract_text(body)

        code, status, message, details = _error_fields(response)
        if code == 429:
            delay = _retry_delay_seconds(details, message)
            daily = _is_daily_quota(f"{message} {json.dumps(details, default=str)}")
            if not daily and not last_attempt and delay is not None and delay <= MAX_RETRY_WAIT_SECONDS:
                logger.warning("Gemini quota hit; retrying in %.1fs (attempt %d)", delay, attempt)
                time.sleep(delay + 0.5)
                continue
            raise LLMRateLimited(
                f"Gemini quota exhausted ({'daily' if daily else 'per-minute'}): {message}",
                retry_after=delay, daily=daily,
            )
        if code in (500, 502, 503, 504) and not last_attempt:
            logger.warning("Gemini %s; retrying (attempt %d)", code, attempt)
            time.sleep(2 ** attempt)
            continue

        detail = f"Gemini API error {code} {status}: {message}"
        if code in (500, 502, 503, 504):  # the retries above are exhausted
            raise LLMUnavailable(detail)
        if code in (401, 403, 404) or (code == 400 and "api key" in message.lower()):
            # bad/unauthorized key, API not enabled, or a model/endpoint that doesn't exist (404)
            raise LLMConfigError(
                f"{detail} (check GEMINI_API_KEY and GEMINI_MODEL"
                + (", and that GEMINI_BASE_URL points at a Gemini-compatible base" if _custom_endpoint() else "")
                + ")"
            )
        raise LLMError(detail)

    raise LLMError("Gemini request failed")  # pragma: no cover - loop always returns or raises


def parse_json_object(raw: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Return (data, []) on success or (None, [problem]) in a form that can be fed back to the model."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, ["the response was not valid JSON (it may have been cut off); return the complete JSON object"]
    if not isinstance(data, dict):
        return None, ["the response must be a single JSON object"]
    return data, []
