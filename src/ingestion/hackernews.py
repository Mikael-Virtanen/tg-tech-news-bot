"""
Hacker News front page via the Algolia HN Search API (one request, no scraping).

    GET https://hn.algolia.com/api/v1/search?tags=front_page

Each hit carries title, url (absent for Ask HN), points and objectID. The linked article is
downloaded (SSRF-guarded, see base.fetch_page) so the LLM works from real text rather than a
headline; items whose text can't be obtained are skipped instead of being summarized blind.
"""
from __future__ import annotations

import logging
from collections.abc import Container
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

from src.ingestion.base import (
    API_TIMEOUT,
    MIN_TEXT_CHARS,
    Source,
    SourceError,
    extract_main_text,
    fetch_page,
    html_to_text,
    make_session,
)
from src.llm.prompts import ARTICLE_SOURCE_CHARS
from src.models import NewsItem

API_URL = "https://hn.algolia.com/api/v1/search"
DISCUSSION_URL = "https://news.ycombinator.com/item?id={id}"

logger = logging.getLogger(__name__)


class HackerNewsSource(Source):
    name = "hackernews"

    def __init__(
        self,
        session: requests.Session | None = None,
        *,
        min_points: int = 100,
        min_text_chars: int = MIN_TEXT_CHARS,
        max_text_chars: int = ARTICLE_SOURCE_CHARS,
        max_workers: int = 5,
    ) -> None:
        self._session = session or make_session()
        self._min_points = min_points
        self._min_text_chars = min_text_chars
        self._max_text_chars = max_text_chars
        self._max_workers = max_workers

    # ------------------------------------------------------------------ #
    def fetch(self, limit: int = 20, skip_ids: Container[str] = frozenset()) -> list[NewsItem]:
        hits = [
            h for h in self._front_page()
            if h.get("title")
            and (h.get("points") or 0) >= self._min_points
            and f"{self.name}:{h['objectID']}" not in skip_ids  # before any article download
        ]
        hits.sort(key=lambda h: h["points"], reverse=True)
        candidates = hits[: limit * 2]  # headroom: some pages will be paywalled, PDFs, JS-only, ...

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            texts = list(pool.map(self._text_for, candidates))

        items: list[NewsItem] = []
        for hit, text in zip(candidates, texts):
            if len(text) < self._min_text_chars:
                logger.info("Skipping HN %s: only %d chars of text", hit["objectID"], len(text))
                continue
            items.append(
                NewsItem(
                    id=f"hackernews:{hit['objectID']}",
                    source=self.name,
                    title=hit["title"].strip(),
                    url=hit.get("url") or DISCUSSION_URL.format(id=hit["objectID"]),
                    text=text,
                    score=int(hit["points"]),
                )
            )
            if len(items) >= limit:
                break
        return items

    # ------------------------------------------------------------------ #
    def _front_page(self) -> list[dict[str, Any]]:
        try:
            resp = self._session.get(
                API_URL, params={"tags": "front_page", "hitsPerPage": 50}, timeout=API_TIMEOUT
            )
            resp.raise_for_status()
            return list(resp.json()["hits"])
        except (requests.RequestException, KeyError, ValueError) as exc:
            raise SourceError(f"Hacker News API failed: {exc}") from exc

    def _text_for(self, hit: dict[str, Any]) -> str:
        """Best available text: linked article, else the post's own text (Ask/Show HN)."""
        try:
            url = hit.get("url")
            if url:
                page = fetch_page(self._session, url)
                if page:
                    text = extract_main_text(page, self._max_text_chars)
                    if len(text) >= self._min_text_chars:
                        return text
            return html_to_text(hit.get("story_text"))[: self._max_text_chars]
        except Exception:  # noqa: BLE001 - one weird page must not break the whole batch
            logger.exception("Text extraction failed for HN %s", hit.get("objectID"))
            return ""
