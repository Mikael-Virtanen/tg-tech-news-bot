"""Plain data containers shared by every layer (no behaviour, no I/O)."""
from __future__ import annotations

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


@dataclass(frozen=True)
class Article:
    """Full Persian article for Telegraph. `body` uses the line syntax understood by
    publishing/telegraph.py::text_to_nodes (## heading, - bullet, --- divider, paragraphs)."""

    title: str
    body: str
