"""
Summarization pipeline: NewsItem -> validated Persian Summary.

    call model -> parse JSON -> normalize -> validate
        |-- valid                      -> return
        |-- violations (attempt 1)     -> retry once, telling the model exactly what it broke
        `-- violations (attempt 2)     -> deterministic fallback (strip / truncate), re-validate

Contract: `summarize` either returns a Summary that satisfies every limit in
llm/prompts.py, or raises SummarizationError. Callers (handlers) should catch it,
log, and skip the item; never post unvalidated model output.

Environment variables and model notes: see llm/client.py.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from src.llm.client import LLMError, Messages, call_model, parse_json_object
from src.llm.prompts import (
    BULLET_COUNT,
    BULLET_MAX,
    EMOJI,
    HASHTAG_COUNT,
    HASHTAG_MAX,
    HOOK_MAX,
    RESPONSE_FORMAT,
    SYSTEM_PROMPT,
    TITLE_MAX,
    build_user_message,
    normalize_summary,
    validate_summary,
)
from src.models import NewsItem, Summary

MAX_ATTEMPTS = 2  # 1 initial call + 1 corrective retry

logger = logging.getLogger(__name__)


class SummarizationError(LLMError):
    """The model refused, never produced usable JSON, or the output can't be repaired."""


# --------------------------------------------------------------------------- #
# OpenAI call
# --------------------------------------------------------------------------- #
def _call_model(messages: Messages) -> str:
    """Single chat-completion call; returns the raw message content."""
    return call_model(messages, RESPONSE_FORMAT, max_completion_tokens=800, timeout=30.0)


# --------------------------------------------------------------------------- #
# Parsing / retry helpers
# --------------------------------------------------------------------------- #
def _parse(raw: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Return (normalized data, []) on success or (None, [problem]) if unusable."""
    data, problems = parse_json_object(raw)
    if data is None:
        return None, problems
    return normalize_summary(data), []


def _retry_message(problems: list[str]) -> str:
    listed = "\n".join(f"- {p}" for p in problems)
    return (
        "Your previous answer broke these rules:\n"
        f"{listed}\n\n"
        "Return the corrected JSON object only. Keep the same facts, shorten wording "
        "instead of dropping key information, and re-count every field before answering."
    )


# --------------------------------------------------------------------------- #
# Last-resort deterministic repair
# --------------------------------------------------------------------------- #
_STRIP_RE = re.compile(r"https?://\S+|<[^>]+>")
_TRAILING_JUNK = " ،,.:;-\u200c"


def _clean_field(text: str) -> str:
    text = EMOJI.sub("", _STRIP_RE.sub("", text))
    return " ".join(text.split())


def _truncate(text: str, limit: int) -> str:
    """Cut to `limit` characters (ellipsis included), preferring a word boundary."""
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    space = cut.rfind(" ")
    if space >= limit * 0.6:  # don't throw away too much just to land on a space
        cut = cut[:space]
    return cut.rstrip(_TRAILING_JUNK) + "…"


def _force_fit(data: dict[str, Any]) -> dict[str, Any]:
    """Strip junk and truncate every field to its limit. Raises if bullets are missing."""
    bullets = [b for b in (_clean_field(b) for b in data["bullets"]) if b]
    if len(bullets) < BULLET_COUNT:
        raise SummarizationError(f"only {len(bullets)} usable bullets after cleanup")

    hashtags = [_clean_field(t)[:HASHTAG_MAX].rstrip("_\u200c") for t in data["hashtags"]]
    return {
        "title": _truncate(_clean_field(data["title"]), TITLE_MAX),
        "hook": _truncate(_clean_field(data["hook"]), HOOK_MAX),
        "bullets": [_truncate(b, BULLET_MAX) for b in bullets[:BULLET_COUNT]],
        "hashtags": [t for t in hashtags if t][:HASHTAG_COUNT],
    }


def _to_summary(data: dict[str, Any]) -> Summary:
    return Summary(
        title=data["title"],
        hook=data["hook"],
        bullets=tuple(data["bullets"]),
        hashtags=tuple(data["hashtags"]),
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def summarize(item: NewsItem) -> Summary:
    """Produce a validated Persian summary for `item`, or raise SummarizationError."""
    messages: Messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(item.title, item.source, item.url, item.text)},
    ]
    latest: dict[str, Any] | None = None  # most recent parseable output, kept for the fallback

    for attempt in range(1, MAX_ATTEMPTS + 1):
        raw = _call_model(messages)
        data, problems = _parse(raw)
        if data is not None:
            latest = data
            problems = validate_summary(data)
        if not problems:
            logger.info("Summary for %s accepted on attempt %d", item.id, attempt)
            return _to_summary(data)  # type: ignore[arg-type]

        logger.warning("Summary for %s failed validation (attempt %d): %s", item.id, attempt, problems)
        if attempt < MAX_ATTEMPTS:
            messages = [
                *messages,
                {"role": "assistant", "content": raw},
                {"role": "user", "content": _retry_message(problems)},
            ]

    if latest is None:
        raise SummarizationError(f"model never returned valid JSON for {item.id}")

    fitted = _force_fit(latest)
    remaining = validate_summary(fitted, require_all_hashtags=False)
    if remaining:
        raise SummarizationError(f"could not repair summary for {item.id}: {remaining}")
    logger.warning("Summary for %s was repaired by the fallback truncation", item.id)
    return _to_summary(fitted)
