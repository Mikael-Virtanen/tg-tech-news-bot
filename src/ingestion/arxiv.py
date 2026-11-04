"""
arXiv via the official Atom API (https://info.arxiv.org/help/api/).

    GET https://export.arxiv.org/api/query?search_query=cat:cs.AI OR cat:cs.CL ...&sortBy=submittedDate

arXiv has no popularity signal, so "newest in the chosen categories" would flood the channel with
~100+ papers a day. Two knobs keep it useful:
    keywords   only papers matching at least one keyword are kept, ranked by how many match
    limit      the handler's per-run cap

Only the abstract is available through the API, so these items produce short articles (the article
prompt is written for exactly that case). arXiv asks for at most one request every 3 seconds
and a single connection; this makes one request per run.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Container, Sequence

import feedparser
import requests

from src.ingestion.base import API_TIMEOUT, MIN_TEXT_CHARS, Source, SourceError, make_session, squash
from src.models import NewsItem

API_URL = "https://export.arxiv.org/api/query"
MAX_FETCH = 100  # arXiv API page size we are willing to pull in one request

_VERSION_RE = re.compile(r"v\d+$")

logger = logging.getLogger(__name__)


def arxiv_id(entry_id: str) -> str:
    """'http://arxiv.org/abs/2401.12345v2' -> '2401.12345' (versions must not create duplicates)."""
    return _VERSION_RE.sub("", entry_id.rsplit("/abs/", 1)[-1])


def parse_feed(
    content: bytes | str,
    *,
    keywords: Sequence[str] = (),
    min_text_chars: int = MIN_TEXT_CHARS,
    skip_ids: Container[str] = frozenset(),
) -> list[NewsItem]:
    feed = feedparser.parse(content)
    if feed.bozo and not feed.entries:
        raise SourceError(f"unparseable arXiv feed: {feed.bozo_exception}")

    # Word-start match: "agent" finds "agents", but "rl" no longer matches "world".
    wanted = [re.compile(rf"\b{re.escape(k.lower())}") for k in keywords if k.strip()]
    items: list[NewsItem] = []
    for entry in feed.entries:
        if f"arxiv:{arxiv_id(entry.id)}" in skip_ids:
            continue
        title = squash(entry.get("title", ""))
        abstract = squash(entry.get("summary", ""))
        if not title or len(abstract) < min_text_chars:
            continue

        haystack = f"{title} {abstract}".lower()
        matches = sum(1 for pattern in wanted if pattern.search(haystack))
        if wanted and matches == 0:
            continue

        authors = [a.get("name", "") for a in entry.get("authors", [])][:6]
        category = (entry.get("arxiv_primary_category") or {}).get("term", "")
        lines = [
            f"Authors: {', '.join(authors)}" if authors else "",
            f"Category: {category}" if category else "",
            f"Abstract: {abstract}",
        ]
        items.append(
            NewsItem(
                id=f"arxiv:{arxiv_id(entry.id)}",
                source="arxiv",
                title=title,
                url=entry.get("link", entry.id).replace("http://", "https://", 1),
                text="\n".join(line for line in lines if line),
                score=matches,
            )
        )

    items.sort(key=lambda i: i.score, reverse=True)  # stable: equal scores keep arXiv's newest-first order
    return items


class ArxivSource(Source):
    name = "arxiv"

    def __init__(
        self,
        session: requests.Session | None = None,
        *,
        categories: Sequence[str] = ("cs.AI", "cs.CL", "cs.LG"),
        keywords: Sequence[str] = (),
        min_text_chars: int = MIN_TEXT_CHARS,
    ) -> None:
        self._session = session or make_session()
        self._categories = tuple(categories)
        self._keywords = tuple(keywords)
        self._min_text_chars = min_text_chars

    def fetch(self, limit: int = 20, skip_ids: Container[str] = frozenset()) -> list[NewsItem]:
        # Pull extra when filtering by keyword, since most papers won't match.
        count = min(MAX_FETCH, limit * 5 if self._keywords else limit * 2)
        params = {
            "search_query": " OR ".join(f"cat:{c}" for c in self._categories),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": 0,
            "max_results": count,
        }
        try:
            resp = self._session.get(API_URL, params=params, timeout=max(API_TIMEOUT, 20.0))
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise SourceError(f"arXiv API failed: {exc}") from exc
        return parse_feed(
            resp.content, keywords=self._keywords, min_text_chars=self._min_text_chars, skip_ids=skip_ids
        )[:limit]
