"""
Telegra.ph publisher for the Tech & AI News bot.

Links to telegra.ph get Instant View in Telegram automatically, so the job here is:

    plain text -> Telegraph Node tree -> createPage -> page URL

API docs: https://telegra.ph/api
Requires: Python 3.10+, `requests`.

Environment variables (used by `publish_article`):
    TELEGRAPH_ACCESS_TOKEN   token returned by `create_account` (store it in SSM / Secrets Manager)
    TELEGRAPH_AUTHOR_NAME    optional, shown under the title (e.g. your channel name)
    TELEGRAPH_AUTHOR_URL     optional, e.g. https://t.me/your_channel
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import requests

API_BASE = "https://api.telegra.ph"
MAX_TITLE_LEN = 256                 # Telegraph: 1-256 characters
MAX_CONTENT_BYTES = 64 * 1024       # Telegraph: ~64 KB per page
MAX_FLOOD_WAIT_SECONDS = 10         # longer waits are raised, not slept (Lambda timeout / billing)

logger = logging.getLogger(__name__)

Node = str | dict[str, Any]


# --------------------------------------------------------------------------- #
# Errors / result type
# --------------------------------------------------------------------------- #
class TelegraphError(RuntimeError):
    """Request failed or the API returned ok=false."""


class TelegraphFloodWait(TelegraphError):
    """Telegraph asked us to slow down for `seconds`."""

    def __init__(self, seconds: int) -> None:
        super().__init__(f"FLOOD_WAIT_{seconds}")
        self.seconds = seconds


class TelegraphContentTooLarge(TelegraphError):
    """Serialized page content exceeds Telegraph's limit."""


@dataclass(frozen=True)
class TelegraphPage:
    url: str
    path: str
    title: str


# --------------------------------------------------------------------------- #
# Text -> Telegraph nodes
# --------------------------------------------------------------------------- #
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$")
_BULLET_RE = re.compile(r"^[-*•]\s+(.+)$")


def text_to_nodes(body: str) -> list[Node]:
    """
    Convert LLM-produced plain text into Telegraph nodes.

    Supported line syntax (one construct per line):
        # Heading / ## Heading   -> <h3>
        ### Subheading           -> <h4>
        - item / * item / • item -> <ul><li> (consecutive lines are grouped)
        ---                      -> <hr>
        anything else            -> <p>
        blank line               -> ends a list

    Telegraph nodes carry text as plain strings, so no HTML escaping is needed.
    """
    nodes: list[Node] = []
    bullets: list[str] = []

    def flush_bullets() -> None:
        if bullets:
            nodes.append(
                {"tag": "ul", "children": [{"tag": "li", "children": [b]} for b in bullets]}
            )
            bullets.clear()

    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            flush_bullets()
            continue
        if line == "---":
            flush_bullets()
            nodes.append({"tag": "hr"})
            continue
        if m := _HEADING_RE.match(line):
            flush_bullets()
            tag = "h3" if len(m.group(1)) <= 2 else "h4"  # Telegraph only supports h3/h4
            nodes.append({"tag": tag, "children": [m.group(2)]})
            continue
        if m := _BULLET_RE.match(line):
            bullets.append(m.group(1))
            continue
        flush_bullets()
        nodes.append({"tag": "p", "children": [line]})

    flush_bullets()
    return nodes


def _source_footer(source_url: str, label: str) -> list[Node]:
    return [
        {"tag": "hr"},
        {
            "tag": "p",
            "children": [f"{label}: ", {"tag": "a", "attrs": {"href": source_url}, "children": [source_url]}],
        },
    ]


# --------------------------------------------------------------------------- #
# Low-level API call
# --------------------------------------------------------------------------- #
def _api_call(
    session: requests.Session,
    method: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    """
    POST to a Telegraph method and return `result`.

    Retries network errors / non-JSON (5xx) responses with exponential backoff and
    short FLOOD_WAIT_n responses. Caveat: if a createPage request succeeds but its
    response is lost, a retry can create a duplicate page. Harmless for a news
    feed; de-duplicate on your side (e.g. DynamoDB conditional write) if it matters.
    """
    url = f"{API_BASE}/{method}"
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.post(url, data=payload, timeout=timeout)
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == max_retries:
                raise TelegraphError(f"{method} failed after {attempt} attempts: {exc}") from exc
            delay = 2 ** (attempt - 1)
            logger.warning("Telegraph %s attempt %d failed (%s); retrying in %ds", method, attempt, exc, delay)
            time.sleep(delay)
            continue

        if data.get("ok"):
            return data["result"]

        error = str(data.get("error", "UNKNOWN_ERROR"))
        if error.startswith("FLOOD_WAIT_"):
            seconds = int(error.rsplit("_", 1)[1])
            if seconds <= MAX_FLOOD_WAIT_SECONDS and attempt < max_retries:
                logger.warning("Telegraph flood wait: sleeping %ds", seconds)
                time.sleep(seconds)
                continue
            raise TelegraphFloodWait(seconds)

        raise TelegraphError(f"{method}: {error}")

    raise TelegraphError(f"{method}: exhausted retries")  # pragma: no cover


# --------------------------------------------------------------------------- #
# Public client
# --------------------------------------------------------------------------- #
class TelegraphClient:
    def __init__(
        self,
        access_token: str,
        *,
        author_name: str | None = None,
        author_url: str | None = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        session: requests.Session | None = None,
    ) -> None:
        if not access_token:
            raise ValueError("access_token is required (run create_account once and store the token)")
        self._token = access_token
        self._author_name = author_name
        self._author_url = author_url
        self._timeout = timeout
        self._max_retries = max_retries
        self._session = session or requests.Session()

    def publish(
        self,
        title: str,
        body: str,
        *,
        source_url: str | None = None,
        source_label: str = "منبع",
    ) -> TelegraphPage:
        """Publish `body` (see `text_to_nodes` for syntax) and return the page."""
        title = " ".join(title.split())[:MAX_TITLE_LEN]
        if not title:
            raise ValueError("title must not be empty")
        if not body.strip():
            raise ValueError("body must not be empty")

        nodes = text_to_nodes(body)
        if source_url:
            if not source_url.startswith(("http://", "https://")):
                raise ValueError("source_url must start with http:// or https://")
            nodes += _source_footer(source_url, source_label)

        content = json.dumps(nodes, ensure_ascii=False, separators=(",", ":"))
        size = len(content.encode("utf-8"))
        if size > MAX_CONTENT_BYTES:
            raise TelegraphContentTooLarge(
                f"content is {size} bytes (limit {MAX_CONTENT_BYTES}); shorten the article or split it"
            )

        payload: dict[str, Any] = {
            "access_token": self._token,
            "title": title,
            "content": content,          # Telegraph expects the Node array as a JSON string
            "return_content": "false",
        }
        if self._author_name:
            payload["author_name"] = self._author_name[:128]
        if self._author_url:
            payload["author_url"] = self._author_url[:512]

        result = _api_call(
            self._session, "createPage", payload,
            timeout=self._timeout, max_retries=self._max_retries,
        )
        page = TelegraphPage(url=result["url"], path=result["path"], title=result["title"])
        logger.info("Published Telegraph page %s", page.url)
        return page


# --------------------------------------------------------------------------- #
# One-time account setup + Lambda-friendly entry point
# --------------------------------------------------------------------------- #
def create_account(short_name: str, author_name: str = "", author_url: str = "") -> str:
    """
    Create a Telegraph account and return its access_token.
    Run ONCE, then store the token in SSM Parameter Store / Secrets Manager.
    """
    payload = {"short_name": short_name[:32], "author_name": author_name[:128], "author_url": author_url[:512]}
    with requests.Session() as session:
        result = _api_call(session, "createAccount", payload, timeout=10.0, max_retries=3)
    return result["access_token"]


@lru_cache(maxsize=1)
def _client_from_env() -> TelegraphClient:
    # Cached so warm Lambda invocations reuse the connection pool.
    return TelegraphClient(
        os.environ["TELEGRAPH_ACCESS_TOKEN"],
        author_name=os.environ.get("TELEGRAPH_AUTHOR_NAME"),
        author_url=os.environ.get("TELEGRAPH_AUTHOR_URL"),
    )


def publish_article(title: str, body: str, source_url: str | None = None) -> str:
    """Publish a long article and return the telegra.ph URL (what the handlers call)."""
    return _client_from_env().publish(title, body, source_url=source_url).url


if __name__ == "__main__":
    # python -m src.publishing.telegraph "TechBot" "My Channel" "https://t.me/my_channel"
    import sys

    if len(sys.argv) < 2:
        raise SystemExit('usage: python -m src.publishing.telegraph SHORT_NAME [AUTHOR_NAME] [AUTHOR_URL]')
    print(create_account(*sys.argv[1:4]))
