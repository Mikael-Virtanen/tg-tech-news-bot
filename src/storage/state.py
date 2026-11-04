"""
State store for a GitHub Actions deployment: one JSON file, restored from / saved to the `state`
branch by scripts/state_pull.sh and scripts/state_push.sh.

    .state/news.json   {"version": 1, "items": {"<id>": {record}, ...}}

What is stored, and what deliberately is not
    stored      id, source, title, url, score, status, attempts, summary, telegraph_url, message_id,
                timestamps, last_error. A few hundred bytes per item.
    NOT stored  article/README text. It is only needed for the single run that summarizes the item;
                an item that is not picked up in a run simply shows up again in the next run's
                ingestion if it is still trending. This keeps the file (and the repo) small and
                avoids archiving third-party text.

Statuses
    READY       summarized (and on Telegraph), waiting to be posted or digested
    PUBLISHED   posted as a single message            DIGESTED  included in a daily digest
    FAILED      gave up (MAX_ATTEMPTS reached, or delivery to Telegram is unknown)
    RETRY       failed once or twice; NOT in `known_ids()` so ingestion offers it again

Dedup rule: `known_ids()` = every record except RETRY. Records expire after `ttl_days`, which
also ends their dedup window (a GitHub repo may trend again a month later and be re-posted).

Concurrency: there is exactly one writer at a time because both workflows share a
`concurrency:` group. The store therefore needs no locking; it only needs atomic file writes so a
killed process can never leave a half-written file.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.models import NewsItem, Summary

STATUS_RETRY = "RETRY"
STATUS_READY = "READY"
STATUS_PUBLISHED = "PUBLISHED"
STATUS_DIGESTED = "DIGESTED"
STATUS_FAILED = "FAILED"

FILE_NAME = "news.json"
SCHEMA_VERSION = 1
DEFAULT_TTL_DAYS = 30
MAX_ATTEMPTS = 3

logger = logging.getLogger(__name__)


class StateCorruptError(RuntimeError):
    """The state file exists but cannot be trusted. Never auto-reset it: an empty dedup list
    would make the bot re-post everything."""


@dataclass(frozen=True)
class StoredItem:
    item: NewsItem               # `text` is always empty here
    status: str
    attempts: int = 0
    summary: Summary | None = None
    telegraph_url: str | None = None
    message_id: int | None = None
    last_error: str | None = None
    created_at: int = 0


class StateStore:
    def __init__(
        self,
        directory: str | Path,
        *,
        ttl_days: int = DEFAULT_TTL_DAYS,
        persist: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """`persist=False` (dry runs) keeps all changes in memory only."""
        self._path = Path(directory) / FILE_NAME
        self._ttl_seconds = ttl_days * 86_400
        self._persist = persist
        self._clock = clock
        self._items: dict[str, dict[str, Any]] = self._load()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _load(self) -> dict[str, dict[str, Any]]:
        if not self._path.exists():
            logger.info("No state file at %s: starting empty", self._path)
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if data["version"] != SCHEMA_VERSION or not isinstance(data["items"], dict):
                raise ValueError(f"unsupported schema version {data['version']!r}")
            return data["items"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise StateCorruptError(f"cannot read {self._path}: {exc}") from exc

    def save(self) -> None:
        """Atomic write: temp file in the same directory, then rename over the old file."""
        if not self._persist:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"version": SCHEMA_VERSION, "items": self._items}, ensure_ascii=False, sort_keys=True, indent=1
        )
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=".news-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self._path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def known_ids(self) -> frozenset[str]:
        """Ids ingestion must not fetch again (everything except items awaiting a retry)."""
        return frozenset(i for i, r in self._items.items() if r["status"] != STATUS_RETRY)

    def get(self, item_id: str) -> StoredItem | None:
        record = self._items.get(item_id)
        return self._to_stored(item_id, record) if record else None

    def ready(self, *, max_age_hours: float | None = None) -> list[StoredItem]:
        """READY items, highest score first; optionally only those created within `max_age_hours`."""
        cutoff = self._clock() - max_age_hours * 3600 if max_age_hours is not None else None
        found = [
            self._to_stored(i, r)
            for i, r in self._items.items()
            if r["status"] == STATUS_READY and (cutoff is None or r["created_at"] >= cutoff)
        ]
        return sorted(found, key=lambda s: (s.item.score, s.created_at), reverse=True)

    def __len__(self) -> int:
        return len(self._items)

    # ------------------------------------------------------------------ #
    # Updates (in memory; call save() to persist)
    # ------------------------------------------------------------------ #
    def record_ready(self, item: NewsItem, summary: Summary, telegraph_url: str | None = None) -> None:
        now = int(self._clock())
        previous = self._items.get(item.id, {})
        self._items[item.id] = {
            "source": item.source,
            "title": item.title,
            "url": item.url,
            "score": int(item.score),
            "status": STATUS_READY,
            "attempts": previous.get("attempts", 0),
            "created_at": previous.get("created_at", now),
            "updated_at": now,
            "summary": {
                "title": summary.title,
                "hook": summary.hook,
                "bullets": list(summary.bullets),
                "hashtags": list(summary.hashtags),
            },
            "telegraph_url": telegraph_url,
        }

    def mark_published(self, item_id: str, message_id: int, telegraph_url: str | None = None) -> bool:
        record = self._items.get(item_id)
        if record is None or record["status"] != STATUS_READY:
            return False  # never publish twice, never publish something that isn't ready
        record.update(status=STATUS_PUBLISHED, message_id=int(message_id), updated_at=int(self._clock()))
        if telegraph_url:
            record["telegraph_url"] = telegraph_url
        return True

    def mark_digested(self, item_ids: Iterable[str]) -> int:
        """READY -> DIGESTED; returns how many items actually changed."""
        now = int(self._clock())
        changed = 0
        for item_id in item_ids:
            record = self._items.get(item_id)
            if record and record["status"] == STATUS_READY:
                record.update(status=STATUS_DIGESTED, updated_at=now)
                changed += 1
        return changed

    def record_failure(self, item: NewsItem, reason: str) -> str:
        """The item itself caused a failure (bad LLM output, ...). Returns "retry" or "failed"."""
        now = int(self._clock())
        record = self._items.get(item.id) or {
            "source": item.source, "title": item.title, "url": item.url,
            "score": int(item.score), "attempts": 0, "created_at": now,
        }
        record["attempts"] = record.get("attempts", 0) + 1
        record["status"] = STATUS_FAILED if record["attempts"] >= MAX_ATTEMPTS else STATUS_RETRY
        record.update(last_error=reason[:500], updated_at=now)
        self._items[item.id] = record
        return "failed" if record["status"] == STATUS_FAILED else "retry"

    def mark_failed(self, item_id: str, reason: str) -> bool:
        """Terminal failure with no retry (e.g. Telegram may or may not have delivered the post)."""
        record = self._items.get(item_id)
        if record is None:
            return False
        record.update(status=STATUS_FAILED, last_error=reason[:500], updated_at=int(self._clock()))
        return True

    def prune(self) -> int:
        """Drop records older than the TTL (this is what ends their dedup window)."""
        cutoff = self._clock() - self._ttl_seconds
        stale = [i for i, r in self._items.items() if r.get("updated_at", r["created_at"]) < cutoff]
        for item_id in stale:
            del self._items[item_id]
        return len(stale)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_stored(item_id: str, record: dict[str, Any]) -> StoredItem:
        raw = record.get("summary")
        summary = (
            Summary(raw["title"], raw["hook"], tuple(raw["bullets"]), tuple(raw["hashtags"])) if raw else None
        )
        return StoredItem(
            item=NewsItem(
                id=item_id, source=record["source"], title=record["title"],
                url=record["url"], text="", score=int(record["score"]),
            ),
            status=record["status"],
            attempts=int(record.get("attempts", 0)),
            summary=summary,
            telegraph_url=record.get("telegraph_url"),
            message_id=record.get("message_id"),
            last_error=record.get("last_error"),
            created_at=int(record.get("created_at", 0)),
        )
