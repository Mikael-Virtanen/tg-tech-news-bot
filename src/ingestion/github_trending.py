"""
GitHub Trending.

GitHub has no official "trending" API, so this scrapes https://github.com/trending (selectors
verified against the live page) and, if the markup ever changes and nothing parses, falls back to
the official Search API ("repositories created in the last 7 days, most starred") so the pipeline
degrades instead of going silent.

README text is fetched from raw.githubusercontent.com first: it is not subject to the REST API's
60-requests/hour/IP limit for anonymous callers, which shared Lambda IPs exhaust almost immediately
(verified: the anonymous API quota was already spent when this was tested). The REST API
(`GET /repos/{owner}/{repo}/readme`) is only the fallback for unusual README names; set GITHUB_TOKEN
(a free token with no scopes) to give that fallback 5,000 requests/hour.
"""
from __future__ import annotations

import logging
import os
import re
from collections.abc import Container
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import requests
from bs4 import BeautifulSoup

from src.ingestion.base import API_TIMEOUT, Source, SourceError, make_session, squash
from src.llm.prompts import ARTICLE_SOURCE_CHARS
from src.models import NewsItem

TRENDING_URL = "https://github.com/trending"
API_URL = "https://api.github.com"
README_FETCH_CHARS = 30_000  # raw README is cleaned before being truncated to the LLM budget
RAW_URL = "https://raw.githubusercontent.com/{full_name}/HEAD/{filename}"
README_NAMES = ("README.md", "readme.md", "README.rst", "README.markdown", "README")

_STARS_RE = re.compile(r"([\d,]+)\s+stars?\s+(?:today|this week|this month)", re.IGNORECASE)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_BADGE_LINE_RE = re.compile(r"^\s*\[!\[.*$", re.MULTILINE)
_BLANK_RUN_RE = re.compile(r"\n{3,}")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrendingRepo:
    full_name: str           # "owner/repo"
    description: str
    language: str
    stars_today: int         # 0 when unknown (Search-API fallback)
    total_stars: int


def parse_trending(html: bytes | str) -> list[TrendingRepo]:
    """Parse the trending page. Selectors: article.Box-row > h2 a (name), p (description),
    [itemprop=programmingLanguage], and the 'N stars today' text (matched by regex, so
    it survives CSS-class churn)."""
    soup = BeautifulSoup(html, "html.parser")
    repos: list[TrendingRepo] = []
    for row in soup.select("article.Box-row"):
        link = row.select_one("h2 a[href]")
        if link is None:
            continue
        full_name = str(link["href"]).strip("/")
        if full_name.count("/") != 1:
            continue
        description_el = row.find("p")
        language_el = row.select_one('[itemprop="programmingLanguage"]')
        stars_today = _STARS_RE.search(squash(row.get_text(" ")))
        total_el = row.select_one('a[href$="/stargazers"]')
        repos.append(
            TrendingRepo(
                full_name=full_name,
                description=squash(description_el.get_text()) if description_el else "",
                language=squash(language_el.get_text()) if language_el else "",
                stars_today=int(stars_today.group(1).replace(",", "")) if stars_today else 0,
                total_stars=int(re.sub(r"\D", "", total_el.get_text()) or 0) if total_el else 0,
            )
        )
    return repos


def clean_readme(markdown: str, max_chars: int = ARTICLE_SOURCE_CHARS) -> str:
    """Strip badges, images, raw HTML and link targets so the LLM sees prose, not markup."""
    text = _BADGE_LINE_RE.sub("", markdown)
    text = _IMAGE_RE.sub("", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _HTML_TAG_RE.sub("", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = _BLANK_RUN_RE.sub("\n\n", text).strip()
    return text[:max_chars]


class GitHubTrendingSource(Source):
    name = "github"

    def __init__(
        self,
        session: requests.Session | None = None,
        *,
        since: str = "daily",              # daily | weekly | monthly
        language: str | None = None,       # e.g. "python"; None = all languages
        token: str | None = None,
        min_text_chars: int = 100,
        fetch_readme: bool = True,
    ) -> None:
        self._session = session or make_session()
        self._since = since
        self._language = language
        self._token = token if token is not None else os.environ.get("GITHUB_TOKEN")
        self._min_text_chars = min_text_chars
        self._fetch_readme = fetch_readme

    # ------------------------------------------------------------------ #
    def fetch(self, limit: int = 20, skip_ids: Container[str] = frozenset()) -> list[NewsItem]:
        repos = self._trending()
        if not repos:
            logger.warning("GitHub trending page yielded no repos (markup changed?); using Search API fallback")
            repos = self._search_fallback(limit * 2)
        if not repos:
            raise SourceError("no repositories found via trending page or Search API")

        repos = [r for r in repos if f"{self.name}:{r.full_name.lower()}" not in skip_ids]  # before any README download
        repos.sort(key=lambda r: (r.stars_today, r.total_stars), reverse=True)
        items: list[NewsItem] = []
        for repo in repos[: limit * 2]:
            text = self._text_for(repo)
            if len(text) < self._min_text_chars:
                logger.info("Skipping %s: only %d chars of text", repo.full_name, len(text))
                continue
            title = f"{repo.full_name}: {repo.description}" if repo.description else repo.full_name
            items.append(
                NewsItem(
                    id=f"github:{repo.full_name.lower()}",
                    source=self.name,
                    title=title[:200],
                    url=f"https://github.com/{repo.full_name}",
                    text=text,
                    score=repo.stars_today or repo.total_stars,
                )
            )
            if len(items) >= limit:
                break
        return items

    # ------------------------------------------------------------------ #
    def _trending(self) -> list[TrendingRepo]:
        path = f"{TRENDING_URL}/{self._language}" if self._language else TRENDING_URL
        try:
            resp = self._session.get(path, params={"since": self._since}, timeout=API_TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("GitHub trending page failed: %s", exc)
            return []
        return parse_trending(resp.content)

    def _search_fallback(self, count: int) -> list[TrendingRepo]:
        since = (date.today() - timedelta(days=7)).isoformat()
        query = f"created:>{since}" + (f" language:{self._language}" if self._language else "")
        try:
            resp = self._session.get(
                f"{API_URL}/search/repositories",
                params={"q": query, "sort": "stars", "order": "desc", "per_page": min(count, 50)},
                headers=self._api_headers(),
                timeout=API_TIMEOUT,
            )
            resp.raise_for_status()
            data: list[dict[str, Any]] = resp.json()["items"]
        except (requests.RequestException, KeyError, ValueError) as exc:
            logger.warning("GitHub Search API fallback failed: %s", exc)
            return []
        return [
            TrendingRepo(
                full_name=r["full_name"],
                description=squash(r.get("description") or ""),
                language=r.get("language") or "",
                stars_today=0,
                total_stars=int(r.get("stargazers_count") or 0),
            )
            for r in data
        ]

    def _api_headers(self, accept: str = "application/vnd.github+json") -> dict[str, str]:
        headers = {"Accept": accept, "X-GitHub-Api-Version": "2022-11-28"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _readme(self, full_name: str) -> str:
        text = self._readme_raw(full_name) or self._readme_api(full_name)
        return clean_readme(text[:README_FETCH_CHARS]) if text else ""

    def _readme_raw(self, full_name: str) -> str:
        """Try the usual file names on the CDN host; a 404 means "try the next name"."""
        for filename in README_NAMES:
            try:
                resp = self._session.get(RAW_URL.format(full_name=full_name, filename=filename), timeout=API_TIMEOUT)
            except requests.RequestException as exc:
                logger.info("raw README for %s failed: %s", full_name, exc)
                return ""
            if resp.status_code == 200:
                return resp.text
            if resp.status_code != 404:  # throttled or erroring: stop hammering, let the API fallback try
                logger.info("raw README for %s: HTTP %s", full_name, resp.status_code)
                return ""
        return ""

    def _readme_api(self, full_name: str) -> str:
        try:
            resp = self._session.get(
                f"{API_URL}/repos/{full_name}/readme",
                headers=self._api_headers("application/vnd.github.raw+json"),
                timeout=API_TIMEOUT,
            )
        except requests.RequestException as exc:
            logger.info("API README for %s failed: %s", full_name, exc)
            return ""
        if resp.status_code != 200:
            logger.info("README for %s unavailable via API (HTTP %s)", full_name, resp.status_code)
            return ""
        return resp.text

    def _text_for(self, repo: TrendingRepo) -> str:
        parts = [repo.description]
        if repo.language:
            parts.append(f"Primary language: {repo.language}")
        if self._fetch_readme:
            parts.append(self._readme(repo.full_name))
        return "\n\n".join(p for p in parts if p).strip()
