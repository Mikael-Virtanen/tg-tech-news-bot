"""Pipeline behaviour with every collaborator faked: no network, no API keys, no sleeping."""
import dataclasses

import pytest

from src.config import Config
from src.ingestion.base import Source
from src.llm.client import LLMError, LLMRateLimited
from src.models import Article, NewsItem, Summary
from src.pipeline import Pipeline, interleave, pick_items
from src.publishing.telegram import TelegramError
from src.publishing.telegraph import TelegraphError
from src.storage.state import StateStore

T0 = 1_800_000_000


class Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


def item(source, n, score):
    return NewsItem(f"{source}:{n}", source, f"{source} story {n}", f"https://example.com/{source}/{n}", "body text", score)


class FakeSource(Source):
    def __init__(self, name, items):
        self.name, self._items, self.skip_history = name, items, []

    def fetch(self, limit=20, skip_ids=frozenset()):
        self.skip_history.append(set(skip_ids))
        return [i for i in self._items if i.id not in skip_ids][:limit]


class FakeTelegram:
    """`errors` is consumed one entry per send_message call: an exception to raise, or None for success."""

    def __init__(self, errors=()):
        self.errors, self.sent = list(errors), []

    def send_message(self, text, *, reply_markup=None, preview_url=None, silent=False):
        error = self.errors.pop(0) if self.errors else None
        if error:
            raise error
        self.sent.append((text, reply_markup))
        return 1000 + len(self.sent)


class Fakes:
    """Stand-ins for the LLM and Telegraph calls, recording what they were asked to do."""

    def __init__(self):
        self.summaries, self.articles, self.pages = [], [], []
        self.summarize_errors, self.article_error, self.on_summarize = {}, None, None

    def summarize(self, it):
        self.summaries.append(it.id)
        if self.on_summarize:
            self.on_summarize()
        error = self.summarize_errors.get(it.id)
        if error:
            raise error() if isinstance(error, type) else error
        return Summary(f"عنوان {it.id}", "یک جمله.", ("الف", "ب", "ج"), ("AI", "LLM", "ML"))

    def write_article(self, it):
        self.articles.append(it.id)
        if self.article_error:
            raise self.article_error
        return Article(f"مقاله {it.id}", "متن مقاله")

    def publish_article(self, title, body, source_url=None):
        self.pages.append(title)
        return f"https://telegra.ph/page-{len(self.pages)}"


def cfg(**overrides):
    base = Config(items_per_source=2, max_posts_per_run=10, write_articles=True)
    return dataclasses.replace(base, **overrides)


def build(tmp_path, sources, *, config=None, telegram=None, fakes=None, clock=None, persist=True):
    fakes = fakes or Fakes()
    config = config or cfg()
    store = StateStore(tmp_path, persist=persist, clock=lambda: T0)
    telegram = telegram if telegram is not None else (None if config.dry_run else FakeTelegram())
    pipeline = Pipeline(
        store, telegram, config, sources=sources,
        summarize=fakes.summarize, write_article=fakes.write_article, publish_article=fakes.publish_article,
        clock=clock or Clock(), today=lambda: "2026-11-03",
    )
    return pipeline, store, telegram, fakes


def reload(tmp_path):
    return StateStore(tmp_path, clock=lambda: T0)


# ------------------------------ selection ----------------------------------- #
def test_interleave_and_pick_items_give_every_source_a_turn():
    assert interleave([[1, 2, 3], ["a"], ["x", "y"]]) == [1, "a", "x", 2, "y", 3]
    items = [item("hn", 1, 10), item("hn", 2, 50), item("hn", 3, 30), item("gh", 1, 5), item("gh", 2, 9)]
    # scores are only comparable within a source: gh's 9 must not lose to hn's 50
    assert [i.id for i in pick_items(items, 2)] == ["hn:2", "gh:2", "hn:3", "gh:1"]


# ------------------------------ single mode --------------------------------- #
def test_single_mode_posts_the_best_items_and_persists_everything(tmp_path):
    hn = FakeSource("hn", [item("hn", 1, 10), item("hn", 2, 50), item("hn", 3, 30)])
    gh = FakeSource("gh", [item("gh", 1, 5), item("gh", 2, 9)])
    pipeline, store, telegram, fakes = build(tmp_path, [hn, gh])

    stats = pipeline.run()

    assert (stats.summarized, stats.published, stats.failed, stats.stopped) == (4, 4, 0, None)
    assert fakes.summaries == ["hn:2", "gh:2", "hn:3", "gh:1"]
    first_text, first_markup = telegram.sent[0]
    assert "عنوان hn:2" in first_text and "<b>" in first_text
    assert [b["url"] for b in first_markup["inline_keyboard"][0]] == ["https://telegra.ph/page-1", "https://example.com/hn/2"]
    saved = reload(tmp_path)
    assert {i: saved.get(i).status for i in ("hn:2", "gh:2", "hn:3", "gh:1")} == dict.fromkeys(("hn:2", "gh:2", "hn:3", "gh:1"), "PUBLISHED")
    assert saved.get("hn:2").message_id == 1001 and saved.get("hn:1") is None  # unpicked items leave no trace


def test_the_same_story_from_two_sources_is_posted_once(tmp_path):
    """Ids are per source; the same link on HN and GitHub must still produce one post."""
    hn = FakeSource("hn", [NewsItem("hn:1", "hn", "story", "https://example.com/dup", "body", 50)])
    gh = FakeSource("gh", [NewsItem("gh:1", "gh", "same story", "http://www.example.com/dup?utm_source=x", "body", 5)])
    pipeline, store, telegram, fakes = build(tmp_path, [hn, gh])

    stats = pipeline.run()

    assert stats.selected == 1 and fakes.summaries == ["hn:1"]  # the higher-scored copy wins
    assert store.known_urls() == {"example.com/dup"}


def test_a_url_posted_in_an_earlier_run_is_skipped_even_from_another_source(tmp_path):
    hn = FakeSource("hn", [item("hn", 1, 50)])
    gh = FakeSource("gh", [NewsItem("gh:1", "gh", "story", "https://example.com/hn/1/", "body", 9)])
    first = build(tmp_path, [hn])
    first[0].run()

    second = build(tmp_path, [gh], telegram=first[2])  # new process, same state on disk
    stats = second[0].run()

    assert stats.selected == 0 and len(first[2].sent) == 1


def test_second_run_never_reposts_and_tells_sources_what_to_skip(tmp_path):
    hn = FakeSource("hn", [item("hn", n, 100 - n) for n in range(1, 6)])
    first = build(tmp_path, [hn], config=cfg(items_per_source=2))
    first[0].run()
    second = build(tmp_path, [hn], config=cfg(items_per_source=2), telegram=first[2])  # new process, same disk + channel
    second[0].run()

    assert hn.skip_history[0] == set() and hn.skip_history[1] == {"hn:1", "hn:2"}
    posted = [text for text, _ in first[2].sent]
    assert len(posted) == 4 and len(set(posted)) == 4  # hn:1..hn:4, each exactly once


def test_post_cap_limits_a_single_run(tmp_path):
    hn = FakeSource("hn", [item("hn", n, 100 - n) for n in range(1, 8)])
    gh = FakeSource("gh", [item("gh", n, 50 - n) for n in range(1, 8)])
    pipeline, _, telegram, _ = build(tmp_path, [hn, gh], config=cfg(items_per_source=5, max_posts_per_run=3))
    stats = pipeline.run()
    assert len(telegram.sent) == 3 and "post cap" in stats.stopped


def test_time_budget_stops_before_starting_another_item(tmp_path):
    clock = Clock()
    fakes = Fakes()
    fakes.on_summarize = lambda: setattr(clock, "now", clock.now + 40)  # each item "takes" 40 s
    hn = FakeSource("hn", [item("hn", n, 100 - n) for n in range(1, 5)])
    pipeline, _, telegram, _ = build(tmp_path, [hn], config=cfg(items_per_source=4, max_runtime_seconds=30), fakes=fakes, clock=clock)
    stats = pipeline.run()
    assert len(telegram.sent) == 1 and "time budget" in stats.stopped


# ------------------------------ digest mode --------------------------------- #
def test_digest_mode_collects_quietly_then_posts_one_message(tmp_path):
    hn = FakeSource("hn", [item("hn", 1, 50), item("hn", 2, 40)])
    gh = FakeSource("gh", [item("gh", 1, 9)])
    pipeline, store, telegram, _ = build(tmp_path, [hn, gh], config=cfg(publish_mode="digest"))

    run_stats = pipeline.run()
    assert run_stats.summarized == 3 and telegram.sent == []  # nothing posted yet
    assert {s.item.id for s in store.ready()} == {"hn:1", "hn:2", "gh:1"}

    digest_stats = pipeline.digest()
    assert digest_stats.digested == 3 and len(telegram.sent) == 1
    text = telegram.sent[0][0]
    assert text.count("<a href=") == 3 and "2026-11-03" in text and "page-" in text  # links point at Telegraph
    assert store.ready() == []
    assert pipeline.digest().digested == 0 and len(telegram.sent) == 1  # a second digest has nothing to say


def test_digest_honours_size_and_age_limits(tmp_path):
    clock_value = [T0]
    store = StateStore(tmp_path, clock=lambda: clock_value[0])
    pipeline = Pipeline(store, FakeTelegram(), cfg(digest_size=2, digest_max_age_hours=48), clock=Clock(), today=lambda: "d")
    fakes = Fakes()
    store.record_ready(item("hn", 1, 99), fakes.summarize(item("hn", 1, 99)))   # created now, then aged below
    clock_value[0] += 50 * 3600
    for n, score in ((2, 10), (3, 30), (4, 20)):
        store.record_ready(item("hn", n, score), fakes.summarize(item("hn", n, score)))

    stats = pipeline.digest()
    assert stats.digested == 2
    # hn:3 (30) and hn:4 (20) are the best FRESH items -> digested; hn:2 (10) waits; hn:1 (50 h old) is excluded by age
    assert {s.item.id for s in store.ready()} == {"hn:1", "hn:2"}


def test_digest_telegram_problems(tmp_path):
    def with_error(error):
        pipeline, store, telegram, fakes = build(tmp_path / str(id(error)), [], telegram=FakeTelegram([error]))
        store.record_ready(item("hn", 1, 1), fakes.summarize(item("hn", 1, 1)))
        return pipeline, store

    pipeline, store = with_error(TelegramError("slow down", error_code=429, retry_after=90))
    assert "digest not sent" in pipeline.digest().stopped and len(store.ready()) == 1  # kept for tomorrow

    pipeline, store = with_error(TelegramError("read timeout", delivery_unknown=True))
    assert pipeline.digest().digested == 1 and store.ready() == []  # at-most-once: do not risk a duplicate

    pipeline, _ = with_error(TelegramError("Forbidden", error_code=403))
    with pytest.raises(TelegramError):
        pipeline.digest()


# ------------------------------ failures ------------------------------------ #
def test_quota_exhaustion_stops_cleanly_and_the_item_is_retried_next_run(tmp_path):
    hn = FakeSource("hn", [item("hn", 1, 90), item("hn", 2, 80), item("hn", 3, 70)])
    fakes = Fakes()
    fakes.summarize_errors["hn:2"] = LLMRateLimited("quota", retry_after=60)
    pipeline, store, telegram, _ = build(tmp_path, [hn], config=cfg(items_per_source=3), fakes=fakes)

    stats = pipeline.run()
    assert len(telegram.sent) == 1 and "quota" in stats.stopped.lower()
    assert store.get("hn:2") is None and "hn:2" not in store.known_ids()  # never recorded: nothing to undo

    del fakes.summarize_errors["hn:2"]
    pipeline.run()
    assert [s for s in fakes.summaries if s == "hn:2"] == ["hn:2", "hn:2"]  # tried twice
    assert store.get("hn:2").status == "PUBLISHED" and store.get("hn:3").status == "PUBLISHED"


def test_a_bad_item_is_retried_then_abandoned_without_blocking_others(tmp_path):
    hn = FakeSource("hn", [item("hn", 1, 99), item("hn", 2, 50)])
    fakes = Fakes()
    fakes.summarize_errors["hn:1"] = LLMError("garbage JSON")
    pipeline, store, telegram, _ = build(tmp_path, [hn], config=cfg(items_per_source=1), fakes=fakes)

    for _ in range(3):
        pipeline.run()
    assert store.get("hn:1").status == "FAILED" and "garbage JSON" in store.get("hn:1").last_error
    assert fakes.summaries.count("hn:1") == 3
    pipeline.run()  # hn:1 is now known, so the best remaining item gets its turn
    assert store.get("hn:2").status == "PUBLISHED" and fakes.summaries.count("hn:1") == 3


def test_failed_article_degrades_to_a_summary_post(tmp_path):
    for error in (LLMError("bad article"), TelegraphError("FLOOD_WAIT_60")):
        fakes = Fakes()
        fakes.article_error = error
        pipeline, store, telegram, _ = build(tmp_path / type(error).__name__, [FakeSource("hn", [item("hn", 1, 5)])], fakes=fakes)
        pipeline.run()
        text, markup = telegram.sent[0]
        assert [b["text"] for b in markup["inline_keyboard"][0]] == ["🔗 منبع خبر"]  # only the source button
        assert store.get("hn:1").telegraph_url is None and store.get("hn:1").status == "PUBLISHED"


def test_articles_can_be_switched_off(tmp_path):
    pipeline, _, telegram, fakes = build(tmp_path, [FakeSource("hn", [item("hn", 1, 5)])], config=cfg(write_articles=False))
    pipeline.run()
    assert fakes.articles == [] and len(telegram.sent) == 1


def test_article_quota_error_stops_the_run_without_recording_the_item(tmp_path):
    fakes = Fakes()
    fakes.article_error = LLMRateLimited("quota", daily=True)
    pipeline, store, telegram, _ = build(tmp_path, [FakeSource("hn", [item("hn", 1, 5), item("hn", 2, 4)])], fakes=fakes)
    stats = pipeline.run()
    assert telegram.sent == [] and store.get("hn:1") is None and "daily" in stats.stopped


# ------------------------------ Telegram problems ---------------------------- #
def test_unknown_delivery_is_never_retried_and_never_blocks_the_next_item(tmp_path):
    telegram = FakeTelegram([TelegramError("read timeout", delivery_unknown=True), None])
    hn = FakeSource("hn", [item("hn", 1, 90), item("hn", 2, 80)])
    pipeline, store, _, _ = build(tmp_path, [hn], config=cfg(items_per_source=2), telegram=telegram)
    stats = pipeline.run()
    assert store.get("hn:1").status == "FAILED" and "delivery unknown" in store.get("hn:1").last_error
    assert store.get("hn:2").status == "PUBLISHED" and stats.failed == 1
    pipeline.run()
    assert len(telegram.sent) == 1  # hn:1 was NOT sent again


def test_flood_control_leaves_the_item_for_the_next_run_which_posts_it_first(tmp_path):
    telegram = FakeTelegram([TelegramError("Too Many Requests", error_code=429, retry_after=100)])
    hn = FakeSource("hn", [item("hn", 1, 90), item("hn", 2, 80)])
    pipeline, store, _, _ = build(tmp_path, [hn], config=cfg(items_per_source=1), telegram=telegram)

    stats = pipeline.run()
    assert "Telegram rate limit" in stats.stopped and store.get("hn:1").status == "READY" and telegram.sent == []

    pipeline.run()
    assert [store.get(i).status for i in ("hn:1", "hn:2")] == ["PUBLISHED", "PUBLISHED"]
    assert "hn:1" in telegram.sent[0][0]  # the backlog went out before anything new


def test_rejected_messages_are_marked_failed_not_retried_forever(tmp_path):
    telegram = FakeTelegram([TelegramError("can't parse entities", error_code=400)])
    pipeline, store, _, _ = build(tmp_path, [FakeSource("hn", [item("hn", 1, 5)])], telegram=telegram)
    assert pipeline.run().failed == 1 and store.get("hn:1").status == "FAILED"
    pipeline.run()
    assert telegram.sent == []


def test_fatal_telegram_error_fails_the_run_but_the_work_done_is_saved(tmp_path):
    telegram = FakeTelegram([TelegramError("Forbidden: bot was kicked", error_code=403)])
    pipeline, _, _, _ = build(tmp_path, [FakeSource("hn", [item("hn", 1, 5)])], telegram=telegram)
    with pytest.raises(TelegramError):
        pipeline.run()
    saved = reload(tmp_path).get("hn:1")
    assert saved.status == "READY" and saved.summary is not None  # the LLM work is not lost; it posts once fixed


def test_leftover_ready_items_are_posted_first_and_counted_against_the_cap(tmp_path):
    fakes = Fakes()
    store = StateStore(tmp_path, clock=lambda: T0)
    for n, score in ((1, 10), (2, 30), (3, 20)):
        store.record_ready(item("hn", n, score), fakes.summarize(item("hn", n, score)))
    store.save()
    pipeline, store, telegram, _ = build(tmp_path, [], config=cfg(max_posts_per_run=2), fakes=fakes)
    pipeline.run()
    assert [store.get(f"hn:{n}").status for n in (1, 2, 3)] == ["READY", "PUBLISHED", "PUBLISHED"]  # best two scores


# ------------------------------ dry run ------------------------------------- #
def test_dry_run_sends_nothing_publishes_nothing_and_saves_nothing(tmp_path, capsys):
    hn = FakeSource("hn", [item("hn", 1, 5)])
    pipeline, store, telegram, fakes = build(tmp_path, [hn], config=cfg(dry_run=True), persist=False)
    stats = pipeline.run()
    out = capsys.readouterr().out
    assert telegram is None and stats.published == 1
    assert "DRY RUN: would post hn:1" in out and "DRY RUN: article for hn:1" in out
    assert fakes.articles == ["hn:1"] and fakes.pages == []  # article text generated for review, never published
    assert not (tmp_path / "news.json").exists()


def test_dry_run_digest_posts_and_marks_nothing(tmp_path, capsys):
    pipeline, store, _, fakes = build(tmp_path, [], config=cfg(dry_run=True), persist=False)
    store.record_ready(item("hn", 1, 5), fakes.summarize(item("hn", 1, 5)))
    assert pipeline.digest().digested == 1
    assert "DRY RUN: would post digest" in capsys.readouterr().out and len(store.ready()) == 1


def test_a_telegram_client_is_mandatory_outside_dry_runs(tmp_path):
    with pytest.raises(ValueError):
        Pipeline(StateStore(tmp_path), None, cfg())


# ------------------------------ configuration errors ------------------------- #
def test_a_config_error_fails_the_run_and_is_never_recorded_against_items(tmp_path):
    """Regression: a missing/invalid API key used to be logged as 'retry' for every item, so three
    scheduled runs later real news items were permanently FAILED while the job stayed green."""
    from src.llm.client import LLMConfigError

    hn = FakeSource("hn", [item("hn", 1, 90), item("hn", 2, 80)])
    fakes = Fakes()
    fakes.summarize_errors["hn:1"] = LLMConfigError("GEMINI_API_KEY is not set")
    pipeline, store, telegram, _ = build(tmp_path, [hn], fakes=fakes)

    for _ in range(5):
        with pytest.raises(LLMConfigError):
            pipeline.run()
    assert len(store) == 0 and store.known_ids() == frozenset() and telegram.sent == []  # nothing recorded, nothing burned


def test_an_invalid_telegraph_token_is_loud_not_silently_degraded(tmp_path):
    from src.llm.client import LLMConfigError

    fakes = Fakes()
    fakes.article_error = TelegraphError("createPage: ACCESS_TOKEN_INVALID")
    pipeline, store, telegram, _ = build(tmp_path, [FakeSource("hn", [item("hn", 1, 5)])], fakes=fakes)
    with pytest.raises(TelegraphError):
        pipeline.run()
    assert telegram.sent == [] and store.get("hn:1") is None

    fakes.article_error = LLMConfigError("model not found")
    with pytest.raises(LLMConfigError):
        pipeline.run()


def test_an_outage_stops_the_run_without_blaming_any_item(tmp_path):
    """Regression: a multi-hour Gemini outage must not walk real items towards FAILED."""
    from src.llm.client import LLMUnavailable

    hn = FakeSource("hn", [item("hn", 1, 90), item("hn", 2, 80)])
    fakes = Fakes()
    fakes.summarize_errors["hn:1"] = LLMUnavailable("Gemini API error 503 UNAVAILABLE")
    pipeline, store, telegram, _ = build(tmp_path, [hn], fakes=fakes)

    for _ in range(5):                                  # five scheduled runs during the outage
        stats = pipeline.run()
        assert "unavailable" in stats.stopped.lower()
    assert len(store) == 0 and telegram.sent == []      # nothing recorded: no attempts burned

    del fakes.summarize_errors["hn:1"]
    pipeline.run()                                      # service is back: everything goes out
    assert store.get("hn:1").status == "PUBLISHED" and store.get("hn:2").status == "PUBLISHED"
