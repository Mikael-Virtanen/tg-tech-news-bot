"""
Thin Google Gemini wrapper shared by the summarizer and the article writer.

The rest of the LLM layer is backend-agnostic: it passes OpenAI-style `messages`
(system / user / assistant) and the OpenAI-style `response_format` dict from
llm/prompts.py. This module is the ONLY place that knows about Gemini, and it
translates both:

    system messages        -> `system_instruction`
    user / assistant turns -> `contents` with roles "user" / "model"
    response_format schema -> `response_mime_type="application/json"` + `response_json_schema`

API choice: this uses `models.generate_content`, which is stateless (the full history is
sent on every call, which is exactly what the corrective-retry loop does) and which
Google documents as fully supported. Google recommends the newer Interactions API for
new projects; it stores conversations server-side by default, so it isn't used here. Moving to it
later only touches `call_model` below.

Environment:
    GEMINI_API_KEY          free key from https://aistudio.google.com/apikey (GOOGLE_API_KEY also works)
    GEMINI_BASE_URL         optional: base URL of a proxy/gateway that speaks the Gemini API
                            (e.g. https://proxy.example.com). Empty = Google's default endpoint.
                            The key you set in GEMINI_API_KEY must be valid for that endpoint.
    GEMINI_MODEL            optional, defaults to DEFAULT_MODEL
    GEMINI_THINKING_LEVEL   optional: minimal | low | medium | high (Gemini 3.x models)
    GEMINI_TEMPERATURE      optional float; overrides the callers' temperature on every model

Free-tier notes:
    - Per-model RPM/RPD limits are not published in the docs; they are shown for your
      project at https://aistudio.google.com/rate-limit. Quota errors surface as `LLMRateLimited`.
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

from google import genai
from google.genai import errors, types

# GA, free tier, thinks "minimal" by default, so the output-token cap isn't eaten by reasoning.
# gemini-1.5-* and gemini-2.0-* have been shut down; verify the current list at
# https://ai.google.dev/gemini-api/docs/deprecations before changing this.
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"

MAX_CALL_ATTEMPTS = 3          # 1 call + 2 retries for transient failures
MAX_RETRY_WAIT_SECONDS = 20.0  # longer quota waits are raised, not slept (Lambda billing/timeouts)
THINKING_HEADROOM_TOKENS = 2048  # added to max_output_tokens when thinking is enabled (it counts toward the cap)

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
        self.retry_after = retry_after  # seconds, if Google said how long
        self.daily = daily              # True if the per-day quota looks exhausted


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
def _base_url() -> str:
    """Custom endpoint (proxy/gateway) or Google's default. Empty string counts as unset."""
    return (os.environ.get("GEMINI_BASE_URL") or "").strip().rstrip("/") or DEFAULT_BASE_URL


@lru_cache(maxsize=1)
def _client() -> genai.Client:
    # Cached across warm Lambda invocations. Retries are handled in `call_model` so that
    # quota errors (which must not be hammered on the free tier) get special treatment.
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise LLMConfigError("GEMINI_API_KEY is not set (create a free key at https://aistudio.google.com/apikey)")
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(base_url=_base_url()),
    )


# --------------------------------------------------------------------------- #
# Request building
# --------------------------------------------------------------------------- #
def _split_messages(messages: Messages) -> tuple[str | None, list[types.Content]]:
    """OpenAI-style messages -> (system_instruction, Gemini contents)."""
    system_parts: list[str] = []
    contents: list[types.Content] = []
    for message in messages:
        role, text = message["role"], message["content"]
        if role == "system":
            system_parts.append(text)
        elif role in ("user", "assistant"):
            contents.append(
                types.Content(role="user" if role == "user" else "model", parts=[types.Part(text=text)])
            )
        else:
            raise ValueError(f"unsupported message role: {role!r}")
    return ("\n\n".join(system_parts) or None), contents


def _json_schema(response_format: dict[str, Any]) -> dict[str, Any]:
    try:
        return response_format["json_schema"]["schema"]
    except (KeyError, TypeError) as exc:
        raise ValueError("response_format must look like {'json_schema': {'schema': {...}}}") from exc


def _build_config(
    system: str | None,
    response_format: dict[str, Any],
    *,
    model: str,
    max_completion_tokens: int,
    timeout: float,
    temperature: float,
) -> types.GenerateContentConfig:
    max_output_tokens = max_completion_tokens
    kwargs: dict[str, Any] = {
        "system_instruction": system,
        "response_mime_type": "application/json",
        "response_json_schema": _json_schema(response_format),
        "http_options": types.HttpOptions(timeout=int(timeout * 1000), base_url=_base_url()),
    }

    # Google recommends leaving temperature at its default (1.0) on Gemini 3.x models, so the
    # callers' value is only applied to older models unless GEMINI_TEMPERATURE says otherwise.
    override = os.environ.get("GEMINI_TEMPERATURE")
    if override:
        kwargs["temperature"] = float(override)
    elif not model.startswith("gemini-3"):
        kwargs["temperature"] = temperature

    level = os.environ.get("GEMINI_THINKING_LEVEL", "").strip().upper()
    if level:
        try:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=types.ThinkingLevel[level])
        except KeyError as exc:
            raise LLMConfigError(f"invalid GEMINI_THINKING_LEVEL {level!r} (use minimal, low, medium or high)") from exc
        if level != "MINIMAL":
            max_output_tokens += THINKING_HEADROOM_TOKENS

    kwargs["max_output_tokens"] = max_output_tokens
    return types.GenerateContentConfig(**kwargs)


# --------------------------------------------------------------------------- #
# Error classification
# --------------------------------------------------------------------------- #
_RETRY_IN_RE = re.compile(r"retry in ([\d.]+)\s*s", re.IGNORECASE)
_RETRY_DELAY_RE = re.compile(r"""retryDelay['"]?\s*:\s*['"]?([\d.]+)\s*s""")


def _retry_delay_seconds(exc: errors.APIError) -> float | None:
    """Extract Google's suggested wait from a 429 (RetryInfo detail or the message text)."""
    blob = f"{exc.message} {exc.details if hasattr(exc, 'details') else ''}"
    for pattern in (_RETRY_DELAY_RE, _RETRY_IN_RE):
        match = pattern.search(blob)
        if match:
            return float(match.group(1))
    return None


def _is_daily_quota(exc: errors.APIError) -> bool:
    # Heuristic: quota IDs for daily limits look like "GenerateRequestsPerDayPerProjectPerModel-FreeTier".
    return "PerDay" in str(exc)


def _is_network_error(exc: Exception) -> bool:
    module = type(exc).__module__
    return isinstance(exc, (OSError, TimeoutError)) or module.startswith(("httpx", "httpcore", "aiohttp"))


def _name(value: Any) -> str:
    return getattr(value, "name", None) or str(value)


# --------------------------------------------------------------------------- #
# Response handling
# --------------------------------------------------------------------------- #
def _extract_text(response: types.GenerateContentResponse) -> str:
    if not response.candidates:
        feedback = getattr(response, "prompt_feedback", None)
        reason = getattr(feedback, "block_reason", None)
        raise LLMError(f"Gemini blocked the prompt (block_reason={_name(reason)})" if reason else "Gemini returned no candidates")

    finish = _name(response.candidates[0].finish_reason)
    text = response.text
    if not text:
        raise LLMError(f"Gemini returned no text (finish_reason={finish})")
    if finish in _WITHHELD_FINISH_REASONS:
        raise LLMError(f"Gemini withheld the response (finish_reason={finish})")
    # finish == MAX_TOKENS returns the partial text on purpose: it fails JSON parsing, which the
    # callers turn into a corrective retry.
    return text


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
    system, contents = _split_messages(messages)
    config = _build_config(
        system, response_format,
        model=model, max_completion_tokens=max_completion_tokens, timeout=timeout, temperature=temperature,
    )
    client = _client()

    for attempt in range(1, MAX_CALL_ATTEMPTS + 1):
        last_attempt = attempt == MAX_CALL_ATTEMPTS
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
        except errors.APIError as exc:
            if exc.code == 429:
                delay = _retry_delay_seconds(exc)
                daily = _is_daily_quota(exc)
                if not daily and not last_attempt and delay is not None and delay <= MAX_RETRY_WAIT_SECONDS:
                    logger.warning("Gemini quota hit; retrying in %.1fs (attempt %d)", delay, attempt)
                    time.sleep(delay + 0.5)
                    continue
                raise LLMRateLimited(
                    f"Gemini quota exhausted ({'daily' if daily else 'per-minute'}): {exc.message}",
                    retry_after=delay, daily=daily,
                ) from exc
            if exc.code in (500, 502, 503, 504) and not last_attempt:
                logger.warning("Gemini %s; retrying (attempt %d)", exc.code, attempt)
                time.sleep(2 ** attempt)
                continue
            detail = f"Gemini API error {exc.code} {exc.status}: {exc.message}"
            if exc.code in (500, 502, 503, 504):  # the retries above are exhausted
                raise LLMUnavailable(detail) from exc
            if exc.code in (401, 403, 404) or (exc.code == 400 and "API key" in str(exc.message)):
                # bad/unauthorized key, API not enabled, or a model name that doesn't exist (404)
                raise LLMConfigError(f"{detail} (check GEMINI_API_KEY and GEMINI_MODEL)") from exc
            raise LLMError(detail) from exc
        except Exception as exc:  # noqa: BLE001 - only transport-level errors are retried, the rest re-raised
            if not _is_network_error(exc):
                raise
            if last_attempt:
                raise LLMUnavailable(f"Gemini request failed: {exc}") from exc
            logger.warning("Gemini network error (%s); retrying (attempt %d)", exc, attempt)
            time.sleep(2 ** attempt)
            continue

        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            logger.debug("Gemini usage: prompt=%s output=%s total=%s", usage.prompt_token_count,
                         usage.candidates_token_count, usage.total_token_count)
        return _extract_text(response)

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
