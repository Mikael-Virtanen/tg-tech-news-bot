"""
Ingestion plumbing shared by every source.

    Source.fetch(limit) -> list[NewsItem]      one subclass per origin (HN, GitHub, arXiv)
    make_session()                             requests.Session with UA + retries (no long sleeps)
    fetch_page(session, url)                   SSRF-guarded download of an arbitrary article URL
    extract_main_text(html)                    readable text from a web page, capped for the LLM
    collect_items(sources)                     run all sources; one failing source never sinks the run

Sources decide what is worth posting; they do NOT talk to the database. Deduplication
happens in storage/repository.py so that the rule lives in exactly one place.
"""
from __future__ import annotations

import ipaddress
import logging
import os
import socket
from abc import ABC, abstractmethod
from collections.abc import Container, Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.llm.prompts import ARTICLE_SOURCE_CHARS
from src.models import NewsItem

USER_AGENT = os.environ.get("BOT_USER_AGENT") or "TechNewsBot/1.0 (+https://t.me/your_channel)"  # `or`: "" counts as unset
API_TIMEOUT = 15.0
PAGE_TIMEOUT = 8.0
MAX_PAGE_BYTES = 1_500_000
MAX_REDIRECTS = 3
MIN_TEXT_CHARS = 200  # below this the LLM would be summarizing a title; such items are skipped

logger = logging.getLogger(__name__)


class SourceError(RuntimeError):
    """A source could not produce items (network failure, unexpected response format, ...)."""


class Source(ABC):
    name: str

    @abstractmethod
    def fetch(self, limit: int = 20, skip_ids: Container[str] = frozenset()) -> list[NewsItem]:
        """Return up to `limit` candidate items, best first. Must not touch the state store.

        `skip_ids` are ids the pipeline already knows. Sources must drop them BEFORE spending
        HTTP requests on article/README text; otherwise every scheduled run would re-download the
        same dozens of pages just to throw them away.
        """


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def make_session() -> requests.Session:
    """Session with a descriptive User-Agent and short, bounded retries.

    `respect_retry_after_header=False` on purpose: honouring a 60 s Retry-After would burn
    Lambda time. A throttled request just fails and the item is picked up on the next run.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en"})
    retry = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def is_public_http_url(url: str) -> bool:
    """True only for http(s) URLs on ports 80/443 whose host resolves exclusively to public IPs.

    Hacker News links are attacker-controlled, so the bot must not be tricked into fetching
    localhost, link-local/metadata addresses or private networks (SSRF).
    Residual risk: DNS can change between this check and the request (rebinding); acceptable
    here because the fetched body is only ever used as text for an LLM prompt.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    if parsed.port not in (None, 80, 443):
        return False
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or 443, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError):
        return False
    try:
        return bool(infos) and all(ipaddress.ip_address(i[4][0].split("%")[0]).is_global for i in infos)
    except ValueError:
        return False


def fetch_page(
    session: requests.Session,
    url: str,
    *,
    timeout: float = PAGE_TIMEOUT,
    max_bytes: int = MAX_PAGE_BYTES,
) -> bytes | None:
    """Download an HTML page safely; returns raw bytes (BeautifulSoup detects the charset) or None.

    Redirects are followed manually so every hop is re-validated; the body is read in chunks and
    cut at `max_bytes`; non-text content types (PDF, images, archives) are skipped.
    """
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        if not is_public_http_url(current):
            logger.info("Refusing to fetch non-public or invalid URL: %s", current)
            return None
        try:
            with session.get(current, timeout=timeout, stream=True, allow_redirects=False) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("Location")
                    if not location:
                        return None
                    current = urljoin(current, location)
                    continue
                if resp.status_code != 200:
                    return None
                ctype = resp.headers.get("Content-Type", "").lower()
                if not ("html" in ctype or "xml" in ctype or ctype.startswith("text/")):
                    return None
                declared = resp.headers.get("Content-Length", "")
                if declared.isdigit() and int(declared) > max_bytes:
                    return None
                chunks: list[bytes] = []
                size = 0
                for chunk in resp.iter_content(chunk_size=16_384):
                    size += len(chunk)
                    if size > max_bytes:
                        break  # truncated HTML still parses fine; we only need the readable start
                    chunks.append(chunk)
                return b"".join(chunks)
        except requests.RequestException as exc:
            logger.info("Fetching %s failed: %s", current, exc)
            return None
    return None  # too many redirects


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #
_NOISE_TAGS = ("script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg", "iframe", "button")
_TEXT_TAGS = ["h1", "h2", "h3", "h4", "p", "li", "blockquote"]
_MIN_PARAGRAPH_CHARS = 40  # shorter <p>/<li> are almost always navigation crumbs or captions


def _blocks(container) -> list[str]:
    pieces: list[str] = []
    seen: set[str] = set()
    for el in container.find_all(_TEXT_TAGS):
        if el.find(_TEXT_TAGS):  # wrapper (e.g. <li><p>..</p></li>): its children are collected instead
            continue
        text = " ".join(el.get_text(" ", strip=True).split())
        if not text or text in seen:
            continue
        if not el.name.startswith("h") and len(text) < _MIN_PARAGRAPH_CHARS:
            continue
        seen.add(text)
        pieces.append(text)
    return pieces


def _meta_description(soup: BeautifulSoup) -> str:
    for attrs in ({"property": "og:description"}, {"name": "description"}):
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            return " ".join(str(tag["content"]).split())
    return ""


def _cap(pieces: list[str], max_chars: int) -> str:
    out: list[str] = []
    total = 0
    for piece in pieces:
        if total + len(piece) > max_chars:
            remaining = max_chars - total
            if remaining > 200:  # keep a useful tail of the last paragraph, drop crumbs
                out.append(piece[:remaining].rsplit(" ", 1)[0])
            break
        out.append(piece)
        total += len(piece) + 1
    return "\n".join(out)


def extract_main_text(content: bytes | str, max_chars: int = ARTICLE_SOURCE_CHARS) -> str:
    """Readable text of a web page: headings and paragraphs of <article>/<main>/<body>, boilerplate removed."""
    soup = BeautifulSoup(content, "html.parser")
    description = _meta_description(soup)
    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    body = soup.body or soup
    container = soup.find("article") or soup.find("main") or body
    pieces = _blocks(container)
    if container is not body and sum(map(len, pieces)) < MIN_TEXT_CHARS:
        pieces = _blocks(body)  # a tiny <article> (teaser, comment) shouldn't hide the real content
    if description and not any(description[:60] in p for p in pieces):
        pieces.insert(0, description)
    return _cap(pieces, max_chars)


def html_to_text(fragment: str | None) -> str:
    """Plain text of an HTML fragment (e.g. Hacker News `story_text`)."""
    if not fragment:
        return ""
    return BeautifulSoup(fragment, "html.parser").get_text("\n", strip=True)


# --------------------------------------------------------------------------- #
# Orchestration helper
# --------------------------------------------------------------------------- #
def collect_items(
    sources: Iterable[Source],
    per_source_limit: int = 20,
    skip_ids: Container[str] = frozenset(),
) -> list[NewsItem]:
    """Fetch from every source; a failing source is logged and skipped, never fatal."""
    items: list[NewsItem] = []
    for source in sources:
        try:
            fetched = source.fetch(per_source_limit, skip_ids)
        except Exception:  # noqa: BLE001 - isolation is the whole point of this function
            logger.exception("Source %s failed; continuing with the others", getattr(source, "name", source))
            continue
        logger.info("Source %s returned %d items", source.name, len(fetched))
        items.extend(fetched)
    return items




def squash(text: str) -> str:
    """Collapse whitespace runs (including newlines) to single spaces."""
    return " ".join(text.split())
