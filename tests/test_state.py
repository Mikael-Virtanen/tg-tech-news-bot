import json

import pytest

from src.models import NewsItem, Summary
from src.storage import state as state_mod
from src.storage.state import StateCorruptError, StateStore

T0 = 1_800_000_000
SUMMARY = Summary("عنوان", "یک جمله.", ("الف", "ب", "ج"), ("هوش_مصنوعی", "AI"))


class Clock:
    def __init__(self, now=T0):
        self.now = now

    def __call__(self):
        return self.now


def item(n, score=0, source="hackernews", text="large article body that must never be stored"):
    return NewsItem(f"{source}:{n}", source, f"Title {n}", f"https://example.com/{n}", text, score)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(tmp_path, clock):
    return StateStore(tmp_path, clock=clock)


def test_state_round_trips_through_disk_with_unicode_intact(tmp_path, clock):
    first = StateStore(tmp_path, clock=clock)
    first.record_ready(item(1, 5), SUMMARY, "https://telegra.ph/x")
    first.mark_published("hackernews:1", message_id=77)
    first.save()

    reloaded = StateStore(tmp_path, clock=clock).get("hackernews:1")
    assert reloaded.status == "PUBLISHED" and reloaded.message_id == 77
    assert reloaded.summary == SUMMARY and reloaded.telegraph_url == "https://telegra.ph/x"
    assert "عنوان" in (tmp_path / "news.json").read_text(encoding="utf-8")  # not \u-escaped


def test_article_text_is_never_written_to_disk(tmp_path, store):
    store.record_ready(item(1), SUMMARY)
    store.save()
    raw = (tmp_path / "news.json").read_text(encoding="utf-8")
    assert "must never be stored" not in raw
    assert store.get("hackernews:1").item.text == ""


def test_known_ids_cover_everything_except_pending_retries(store):
    store.record_ready(item(1), SUMMARY)
    store.record_ready(item(2), SUMMARY)
    store.mark_published("hackernews:2", 5)
    store.record_failure(item(3), "bad json")  # first failure -> RETRY
    assert store.known_ids() == {"hackernews:1", "hackernews:2"}  # 3 is offered to ingestion again


def test_failures_retry_then_give_up_and_keep_the_reason(store):
    outcomes = [store.record_failure(item(1), f"error {n}") for n in range(state_mod.MAX_ATTEMPTS)]
    assert outcomes == ["retry", "retry", "failed"]
    record = store.get("hackernews:1")
    assert record.status == "FAILED" and record.attempts == 3 and record.last_error == "error 2"
    assert "hackernews:1" in store.known_ids()  # given up: never fetched again


def test_a_retried_item_that_finally_succeeds_keeps_its_attempt_count(store):
    store.record_failure(item(1), "flaky")
    store.record_ready(item(1), SUMMARY)
    stored = store.get("hackernews:1")
    assert stored.status == "READY" and stored.attempts == 1 and "hackernews:1" in store.known_ids()


def test_mark_published_is_only_allowed_once_and_only_from_ready(store):
    assert store.mark_published("nope:1", 1) is False
    store.record_ready(item(1), SUMMARY)
    assert store.mark_published("hackernews:1", 10) is True
    assert store.mark_published("hackernews:1", 11) is False  # no double publish
    assert store.get("hackernews:1").message_id == 10
    store.record_failure(item(2), "x")
    assert store.mark_published("hackernews:2", 1) is False  # a RETRY record has nothing to publish


def test_ready_is_sorted_by_score_and_filtered_by_age(store, clock):
    store.record_ready(item(1, score=10), SUMMARY)
    clock.now += 3600 * 30
    store.record_ready(item(2, score=500), SUMMARY)
    store.record_ready(item(3, score=50), SUMMARY)
    assert [s.item.id for s in store.ready()] == ["hackernews:2", "hackernews:3", "hackernews:1"]
    assert [s.item.id for s in store.ready(max_age_hours=24)] == ["hackernews:2", "hackernews:3"]  # #1 is 30 h old


def test_mark_digested_only_touches_ready_items(store):
    for n in (1, 2, 3):
        store.record_ready(item(n), SUMMARY)
    store.mark_published("hackernews:3", 1)
    assert store.mark_digested(["hackernews:1", "hackernews:3", "hackernews:99"]) == 1
    assert store.mark_digested(["hackernews:1"]) == 0  # already digested
    assert [s.item.id for s in store.ready()] == ["hackernews:2"]


def test_mark_failed_is_terminal(store):
    store.record_ready(item(1), SUMMARY)
    assert store.mark_failed("hackernews:1", "delivery unknown") is True
    assert store.get("hackernews:1").status == "FAILED" and store.ready() == []
    assert store.mark_failed("missing:1", "x") is False


def test_prune_ends_the_dedup_window_after_the_ttl(tmp_path, clock):
    store = StateStore(tmp_path, ttl_days=30, clock=clock)
    store.record_ready(item(1), SUMMARY)
    clock.now += 29 * 86_400
    store.record_ready(item(2), SUMMARY)
    assert store.prune() == 0
    clock.now += 2 * 86_400  # item 1 is now 31 days old, item 2 is 2 days old
    assert store.prune() == 1
    assert store.known_ids() == {"hackernews:2"}


def test_dry_run_store_never_touches_the_disk(tmp_path, clock):
    store = StateStore(tmp_path, persist=False, clock=clock)
    store.record_ready(item(1), SUMMARY)
    store.save()
    assert not (tmp_path / "news.json").exists()
    assert "hackernews:1" in store.known_ids()  # in-memory behaviour is unchanged


@pytest.mark.parametrize("content", ["{not json", "[]", '{"version": 99, "items": {}}', '{"version": 1, "items": []}'])
def test_corrupt_or_foreign_state_raises_instead_of_resetting(tmp_path, content):
    (tmp_path / "news.json").write_text(content)
    with pytest.raises(StateCorruptError):
        StateStore(tmp_path)


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path, store, monkeypatch):
    store.record_ready(item(1), SUMMARY)
    store.save()
    good = (tmp_path / "news.json").read_text()

    store.record_ready(item(2), SUMMARY)

    def exploding_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(state_mod.os, "replace", exploding_replace)
    with pytest.raises(OSError):
        store.save()
    assert (tmp_path / "news.json").read_text() == good  # the previous file is intact
    assert [p.name for p in tmp_path.iterdir()] == ["news.json"]  # and the temp file was cleaned up
    assert json.loads(good)["version"] == 1
