"""
The bot's workflow, with every collaborator injected so it can be tested without a network.

    run()     fetch -> skip known ids -> pick the best N per source -> summarize -> article on Telegraph
              -> (mode "single") post to the channel, or (mode "digest") keep READY for the digest
    digest()  compile the best READY items into ONE message, post it, mark them DIGESTED

Principles
    - State is saved after EVERY item, so a killed job loses at most the item in flight
      (the workflow's final `if: always()` step then pushes whatever was saved).
    - A problem with one item never aborts the batch. Problems with the world do:
        quota / outage          stop cleanly (LLMRateLimited, LLMUnavailable); nothing is recorded against the
                                item, so it is simply retried next run
        LLMConfigError          raise (bad/missing API key, unknown model): never blamed on an item
        fatal TelegramError     raise, so the run goes red (bad token / bot not admin / wrong chat id)
    - Delivery to Telegram is at-most-once: if it is unknown whether a post arrived (read timeout, 5xx)
      the item is marked FAILED instead of being retried, because a duplicate post is worse than
      a missing one in a news channel.
    - A failed article is not a failed item: the post goes out with the summary and a source button.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import zip_longest
from typing import TypeVar

from src.config import Config
from src.ingestion.base import Source, collect_items
from src.llm.article import write_article as default_write_article
from src.llm.client import LLMConfigError, LLMError, LLMRateLimited, LLMTransientError
from src.llm.summarizer import summarize as default_summarize
from src.models import NewsItem
from src.publishing.telegram import TelegramClient, TelegramError
from src.publishing.telegraph import TelegraphError
from src.publishing.telegraph import publish_article as default_publish_article
from src.publishing.templates import MessageTooLong, build_caption, build_digest, build_reply_markup
from src.storage.state import StateStore, StoredItem

T = TypeVar("T")
logger = logging.getLogger(__name__)


@dataclass
class RunStats:
    fetched: int = 0
    selected: int = 0
    summarized: int = 0
    published: int = 0   # posts sent (or, in a dry run, posts that would have been sent)
    digested: int = 0
    failed: int = 0
    stopped: str | None = None  # why the run ended early, if it did

    def render(self) -> str:
        parts = [f"fetched={self.fetched}", f"selected={self.selected}", f"summarized={self.summarized}",
                 f"published={self.published}", f"digested={self.digested}", f"failed={self.failed}"]
        return ", ".join(parts) + (f" | stopped early: {self.stopped}" if self.stopped else "")


def interleave(groups: Sequence[Sequence[T]]) -> list[T]:
    """Round-robin across groups: [a1, a2], [b1] -> [a1, b1, a2]. Keeps one source from dominating."""
    skip = object()
    return [x for row in zip_longest(*groups, fillvalue=skip) for x in row if x is not skip]


def pick_items(items: Iterable[NewsItem], per_source: int) -> list[NewsItem]:
    """Best `per_source` items from each source (scores are only comparable within a source),
    interleaved so every source gets a turn before any gets a second one."""
    by_source: dict[str, list[NewsItem]] = defaultdict(list)
    for item in items:
        by_source[item.source].append(item)
    return interleave([sorted(g, key=lambda i: i.score, reverse=True)[:per_source] for g in by_source.values()])


def _has_link(url: str | None) -> bool:
    return bool(url) and url.startswith(("http://", "https://"))  # type: ignore[union-attr]


class Pipeline:
    def __init__(
        self,
        store: StateStore,
        telegram: TelegramClient | None,
        cfg: Config,
        *,
        sources: Sequence[Source] = (),
        summarize: Callable = default_summarize,
        write_article: Callable = default_write_article,
        publish_article: Callable = default_publish_article,
        clock: Callable[[], float] = time.monotonic,
        today: Callable[[], str] = lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    ) -> None:
        if telegram is None and not cfg.dry_run:
            raise ValueError("a TelegramClient is required unless dry_run is set")
        self._store, self._telegram, self._cfg, self._sources = store, telegram, cfg, list(sources)
        self._summarize, self._write_article, self._publish_article = summarize, write_article, publish_article
        self._clock, self._today = clock, today

    # ------------------------------------------------------------------ #
    # run
    # ------------------------------------------------------------------ #
    def run(self) -> RunStats:
        cfg, store = self._cfg, self._store
        stats = RunStats()
        started = self._clock()

        pruned = store.prune()
        if pruned:
            logger.info("Pruned %d expired records", pruned)

        if cfg.publish_mode == "single" and not self._publish_backlog(stats):
            store.save()
            return stats

        items = collect_items(self._sources, max(2, cfg.items_per_source * 2), store.known_ids())
        selected = pick_items(items, cfg.items_per_source)
        stats.fetched, stats.selected = len(items), len(selected)
        logger.info("Fetched %d new candidates, processing %d", len(items), len(selected))

        for item in selected:
            if self._clock() - started > cfg.max_runtime_seconds:
                stats.stopped = f"time budget of {cfg.max_runtime_seconds}s reached"
                break
            if cfg.publish_mode == "single" and stats.published >= cfg.max_posts_per_run:
                stats.stopped = f"post cap of {cfg.max_posts_per_run} reached"
                break
            try:
                if not self._process(item, stats):
                    break
            except LLMTransientError as exc:
                if isinstance(exc, LLMRateLimited):
                    stats.stopped = f"Gemini quota exhausted ({'daily' if exc.daily else 'per-minute'})"
                else:
                    stats.stopped = f"Gemini unavailable ({exc})"
                logger.warning("%s; the remaining items stay unrecorded and are retried next run", stats.stopped)
                break
            finally:
                store.save()  # after every item, success or not
        store.save()
        return stats

    def _process(self, item: NewsItem, stats: RunStats) -> bool:
        """Summarize (+ article) one item. Returns False if the whole run should stop."""
        try:
            summary = self._summarize(item)
        except (LLMTransientError, LLMConfigError):
            raise  # the service is broken, not this item: don't record anything against it
        except LLMError as exc:
            outcome = self._store.record_failure(item, f"summarization: {exc}")
            stats.failed += outcome == "failed"
            logger.warning("Summarizing %s failed (%s): %s", item.id, outcome, exc)
            return True

        telegraph_url = self._make_article(item)
        self._store.record_ready(item, summary, telegraph_url)
        self._store.save()
        stats.summarized += 1

        if self._cfg.publish_mode == "single":
            return self._send(self._store.get(item.id), stats)  # type: ignore[arg-type]
        return True

    def _make_article(self, item: NewsItem) -> str | None:
        if not self._cfg.write_articles:
            return None
        try:
            article = self._write_article(item)
            if self._cfg.dry_run:
                print(f"--- DRY RUN: article for {item.id} (not published) ---\n{article.title}\n{article.body[:500]}...\n")
                return None
            return self._publish_article(article.title, article.body, item.url)
        except (LLMTransientError, LLMConfigError):
            raise
        except (LLMError, TelegraphError, ValueError) as exc:
            if isinstance(exc, TelegraphError) and "ACCESS_TOKEN" in str(exc):
                raise  # an invalid Telegraph token would otherwise silently disable every article, forever
            logger.warning("No article for %s (%s); posting the summary with a source link only", item.id, exc)
            return None

    # ------------------------------------------------------------------ #
    # publishing
    # ------------------------------------------------------------------ #
    def _publish_backlog(self, stats: RunStats) -> bool:
        """Post READY items left over from earlier runs (e.g. Telegram was down). False = stop the run."""
        ready = self._store.ready()
        backlog = interleave([sorted(g, key=lambda s: s.item.score, reverse=True) for g in _group(ready).values()])
        for stored in backlog[: self._cfg.max_posts_per_run]:
            if not self._send(stored, stats):
                return False
        return True

    def _send(self, stored: StoredItem, stats: RunStats) -> bool:
        """Post one READY item. Returns False if the run should stop (Telegram is throttling us)."""
        item, summary = stored.item, stored.summary
        assert summary is not None, "READY items always carry a summary"
        try:
            caption = build_caption(item, summary)
        except MessageTooLong as exc:
            self._store.mark_failed(item.id, str(exc))
            stats.failed += 1
            return True
        markup = build_reply_markup(item, stored.telegraph_url)

        if self._cfg.dry_run:
            print(f"--- DRY RUN: would post {item.id} ---\n{caption}\n  buttons: {markup}\n")
            stats.published += 1
            return True

        try:
            message_id = self._telegram.send_message(caption, reply_markup=markup)  # type: ignore[union-attr]
        except TelegramError as exc:
            if exc.fatal:
                raise
            if exc.delivery_unknown:
                logger.warning("Delivery of %s is UNKNOWN (%s). Marked FAILED to avoid a duplicate: check the channel.", item.id, exc)
                self._store.mark_failed(item.id, f"delivery unknown: {exc}")
                stats.failed += 1
            elif exc.error_code == 429:
                stats.stopped = "Telegram rate limit"
                return False  # stays READY: posted first thing next run
            else:
                logger.warning("Telegram rejected %s: %s", item.id, exc)
                self._store.mark_failed(item.id, str(exc))
                stats.failed += 1
            self._store.save()
            return True

        self._store.mark_published(item.id, message_id, stored.telegraph_url)
        self._store.save()
        stats.published += 1
        return True

    # ------------------------------------------------------------------ #
    # digest
    # ------------------------------------------------------------------ #
    def digest(self) -> RunStats:
        cfg, store = self._cfg, self._store
        stats = RunStats()
        ready = store.ready(max_age_hours=cfg.digest_max_age_hours)
        ordered = interleave([sorted(g, key=lambda s: s.item.score, reverse=True) for g in _group(ready).values()])
        entries = [
            (s.item, s.summary, s.telegraph_url)
            for s in ordered[: cfg.digest_size]
            if s.summary is not None and (_has_link(s.telegraph_url) or _has_link(s.item.url))
        ]
        if not entries:
            logger.info("Nothing to put in a digest")
            return stats

        text, included = build_digest(entries, self._today())
        ids = [entry[0].id for entry in entries[:included]]  # build_digest only ever stops at the size limit
        if cfg.dry_run:
            print(f"--- DRY RUN: would post digest of {included} items ---\n{text}\n")
            stats.digested = included
            return stats

        try:
            self._telegram.send_message(text)  # type: ignore[union-attr]
        except TelegramError as exc:
            if exc.fatal:
                raise
            if exc.delivery_unknown:
                logger.warning("Digest delivery is UNKNOWN (%s). Items marked DIGESTED to avoid a duplicate: check the channel.", exc)
            else:
                stats.stopped = f"digest not sent: {exc}"
                return stats  # items stay READY for the next digest
        stats.digested = store.mark_digested(ids)
        store.save()
        return stats


def _group(items: Iterable[StoredItem]) -> dict[str, list[StoredItem]]:
    groups: dict[str, list[StoredItem]] = defaultdict(list)
    for stored in items:
        groups[stored.item.source].append(stored)
    return groups
