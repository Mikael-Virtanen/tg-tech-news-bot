"""Plain data containers shared by every layer (no I/O)."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class NewsItem:
    """One raw item produced by an ingestion source."""

    id: str                 # stable per source, used for de-duplication (e.g. "hackernews:41234567")
    source: str             # "hackernews" | "github" | "arxiv"
    title: str
    url: str                # canonical link; fall back to the discussion page if the item has none
    text: str = ""          # article body / abstract / README excerpt fed to the LLM
    score: int = 0          # source-specific ranking signal (HN points, GitHub stars today, ...)


@dataclass(frozen=True)
class Summary:
    """Validated Persian summary. Every field already satisfies the limits in llm/prompts.py."""

    title: str
    hook: str
    bullets: tuple[str, ...]
    hashtags: tuple[str, ...]


_TRACKING_PARAMS = frozenset(
    {"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src",
     "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content"}
)
_URL_KEY_RE = re.compile(r"^[a-z0-9._-]+")


def normalize_url(url: str | None) -> str:
    """Canonical key for cross-source dedup: scheme-less, www-less, fragment-less, no tracking params.

    "https://www.Example.com/a/?utm_source=x#top" and "http://example.com/a" both become "example.com/a".
    Other query parameters are kept: ?v=... distinguishes two videos, and dropping it would create
    false duplicates.
    """
    if not url:
        return ""
    url = url.strip().lower()
    if "://" in url:
        url = url.split("://", 1)[1]
    url = url.split("#", 1)[0]
    if url.startswith("www."):
        url = url[4:]
    base, sep, query = url.partition("?")
    if sep:
        kept = [p for p in query.split("&") if p and p.split("=", 1)[0] not in _TRACKING_PARAMS]
        url = f"{base}?{'&'.join(kept)}" if kept else base
    url = url.rstrip("/")
    return url if _URL_KEY_RE.match(url) else ""


@dataclass(frozen=True)
class Article:
    """Full Persian article for Telegraph. `body` uses the line syntax understood by
    publishing/telegraph.py::text_to_nodes (## heading, - bullet, --- divider, paragraphs)."""

    title: str
    body: str
