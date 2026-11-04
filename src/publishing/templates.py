"""
Message templates: Summary -> Telegram HTML.

Everything the LLM writes is treated as untrusted text: it is HTML-escaped before it
goes into a template, so a stray "<" or "&" can never break Telegram's parser. Tags
(<b>, <i>, <a>) come only from the templates below.

Length is measured the way Telegram does (tags removed, entities decoded) but in
UTF-16 code units, which over-counts astral emojis by one. That is deliberately
conservative: we'd rather be 2 characters under the limit than get a 400 from the API.
"""
from __future__ import annotations

import html
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from src.llm.prompts import CAPTION_LIMIT
from src.models import NewsItem, Summary

MESSAGE_LIMIT = 4096

SOURCE_EMOJI = {"hackernews": "🟧", "github": "🐙", "arxiv": "📄"}

_TAG_RE = re.compile(r"<[^>]+>")
_NON_WORD_RE = re.compile(r"[^\w]")
_MULTI_UNDERSCORE_RE = re.compile(r"_+")


class MessageTooLong(ValueError):
    """Even the most compact layout does not fit Telegram's limit."""


# --------------------------------------------------------------------------- #
# Branding
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Branding:
    channel_handle: str = "@your_channel"
    footer_emoji: str = "📢"
    bullet_emoji: str = "🔹"
    default_source_emoji: str = "🤖"
    fixed_hashtags: tuple[str, ...] = ("اخبار_تکنولوژی",)  # appended to every post


@lru_cache(maxsize=1)
def default_branding() -> Branding:
    # `or`: GitHub Actions passes an unset `vars.CHANNEL_HANDLE` as an empty string
    return Branding(channel_handle=os.environ.get("CHANNEL_HANDLE") or Branding.channel_handle)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def esc(text: str) -> str:
    """Escape text for Telegram HTML parse mode (only &, <, > need escaping)."""
    return html.escape(text, quote=False)


def telegram_length(text: str) -> int:
    """Visible length of an HTML message, in UTF-16 code units (conservative)."""
    visible = html.unescape(_TAG_RE.sub("", text))
    return len(visible.encode("utf-16-le")) // 2


def format_hashtags(tags: Iterable[str], fixed: Iterable[str] = ()) -> str:
    """
    Turn raw tags into valid, de-duplicated Telegram hashtags.
    Telegram hashtags allow letters, digits and "_" only; ZWNJ and spaces become "_",
    the common convention for Persian hashtags (#هوش_مصنوعی).
    """
    seen: set[str] = set()
    result: list[str] = []
    for raw in (*tags, *fixed):
        tag = _NON_WORD_RE.sub("_", raw.replace("\u200c", "_"))
        tag = _MULTI_UNDERSCORE_RE.sub("_", tag).strip("_")
        if tag and tag.lower() not in seen:
            seen.add(tag.lower())
            result.append(f"#{tag}")
    return " ".join(result)


def _is_http_url(url: str | None) -> bool:
    return bool(url) and url.startswith(("http://", "https://"))  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# Single post
# --------------------------------------------------------------------------- #
def build_caption(item: NewsItem, summary: Summary, branding: Branding | None = None) -> str:
    """
    Build the HTML caption for a single post (<= CAPTION_LIMIT visible characters).

    Layout:
        🟧 <b>Title</b>

        <i>Hook sentence</i>

        🔹 bullet 1
        🔹 bullet 2
        🔹 bullet 3

        #tag1 #tag2 #tag3 #fixed
        📢 @channel

    If a (non-compliant) summary doesn't fit, optional parts are dropped in a fixed
    order (hook, then hashtags) instead of cutting text, so HTML is never left broken.
    """
    brand = branding or default_branding()
    emoji = SOURCE_EMOJI.get(item.source.lower(), brand.default_source_emoji)
    title = f"{emoji} <b>{esc(summary.title)}</b>"
    hook = f"<i>{esc(summary.hook)}</i>"
    bullets = "\n".join(f"{brand.bullet_emoji} {esc(b)}" for b in summary.bullets)
    hashtags = format_hashtags(summary.hashtags, brand.fixed_hashtags)
    footer = f"{brand.footer_emoji} {esc(brand.channel_handle)}"

    layouts = (
        (title, hook, bullets, hashtags, footer),
        (title, bullets, hashtags, footer),   # drop hook
        (title, bullets, footer),             # drop hook + hashtags
    )
    for parts in layouts:
        caption = "\n\n".join(p for p in parts if p)
        if telegram_length(caption) <= CAPTION_LIMIT:
            return caption
    raise MessageTooLong(f"caption exceeds {CAPTION_LIMIT} characters even in compact layout")


def build_reply_markup(item: NewsItem, telegraph_url: str | None = None) -> dict[str, Any] | None:
    """
    Inline keyboard for sendMessage / sendPhoto (`reply_markup=`): "Read full" -> Telegraph
    (Instant View), "Source" -> original link. Returns None if neither URL is usable.
    """
    row: list[dict[str, str]] = []
    if _is_http_url(telegraph_url):
        row.append({"text": "📖 مطالعه کامل", "url": telegraph_url})  # type: ignore[dict-item]
    if _is_http_url(item.url):
        row.append({"text": "🔗 منبع خبر", "url": item.url})
    return {"inline_keyboard": [row]} if row else None


# --------------------------------------------------------------------------- #
# Daily digest
# --------------------------------------------------------------------------- #
DigestEntry = tuple[NewsItem, Summary, "str | None"]  # (item, summary, telegraph_url)


def build_digest(
    entries: Sequence[DigestEntry],
    date_label: str,
    branding: Branding | None = None,
) -> tuple[str, int]:
    """
    Compile top stories into ONE message (<= MESSAGE_LIMIT). Entries are added in order
    until the next one wouldn't fit, so pass them best-first.

    Returns (html_text, number_of_entries_included).
    `date_label` is supplied by the caller so you can pass a Jalali date (e.g. via jdatetime).
    Send with disable_web_page_preview=True, otherwise Telegram renders a preview of the first link.
    """
    brand = branding or default_branding()
    header = f"📰 <b>خلاصه اخبار تکنولوژی و هوش مصنوعی</b>\n🗓 {esc(date_label)}"
    footer = "\n\n".join(
        p for p in (format_hashtags((), brand.fixed_hashtags), f"{brand.footer_emoji} {esc(brand.channel_handle)}") if p
    )

    text = header
    included = 0
    for item, summary, telegraph_url in entries:
        link = telegraph_url if _is_http_url(telegraph_url) else item.url
        if not _is_http_url(link):
            continue
        block = (
            f'{brand.bullet_emoji} <a href="{html.escape(link, quote=True)}"><b>{esc(summary.title)}</b></a>\n'
            f"{esc(summary.hook)}"
        )
        candidate = f"{text}\n\n{block}"
        if telegram_length(f"{candidate}\n\n{footer}") > MESSAGE_LIMIT:
            break
        text = candidate
        included += 1

    if included == 0:
        raise MessageTooLong("no digest entry fits (or none had a usable link)")
    return f"{text}\n\n{footer}", included
