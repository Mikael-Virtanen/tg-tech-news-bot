"""Run with: pytest -q   (from the project root; no network or API key needed)."""
import json
from unittest.mock import patch

import pytest

from src.llm import summarizer
from src.llm.prompts import BULLET_MAX, HASHTAG_MAX, HOOK_MAX, TITLE_MAX
from src.models import NewsItem, Summary
from src.publishing import templates as t

ITEM = NewsItem(id="hackernews:1", source="hackernews", title="Show HN: X", url="https://example.com/x", text="body")


def model_json(**overrides):
    data = {
        "title": "مدل جدید منتشر شد",
        "hook": "یک مدل متن‌باز تازه عرضه شده است.",
        "bullets": ["مدل جدید متن‌باز است", "سرعت استنتاج بهتر شده است", "برای توسعه‌دهندگان مفید است"],
        "hashtags": ["هوش_مصنوعی", "OpenSource", "LLM"],
    }
    data.update(overrides)
    return json.dumps(data, ensure_ascii=False)


# ----------------------------- summarizer ---------------------------------- #
def test_valid_first_attempt():
    with patch.object(summarizer, "_call_model", side_effect=[model_json()]) as call:
        s = summarizer.summarize(ITEM)
    assert call.call_count == 1
    assert len(s.bullets) == 3


def test_retry_feeds_violations_back_to_model():
    too_long = model_json(bullets=["ا" * 200, "ب", "ج"])
    with patch.object(summarizer, "_call_model", side_effect=[too_long, model_json()]) as call:
        s = summarizer.summarize(ITEM)
    assert call.call_count == 2
    retry_prompt = call.call_args_list[1].args[0][-1]["content"]
    assert "bullet 1 is 200 characters" in retry_prompt
    assert s.bullets[0] == "مدل جدید متن‌باز است"


def test_invalid_json_then_valid():
    with patch.object(summarizer, "_call_model", side_effect=['{"title": "cut off', model_json()]):
        assert summarizer.summarize(ITEM).title == "مدل جدید منتشر شد"


def test_fallback_truncates_and_strips_after_two_failures():
    bad = model_json(
        title="ع" * 90,
        hook="جمله " * 40,
        bullets=["کلمه " * 60 + "https://spam.example/a", "ب 🚀", "ج"],
        hashtags=["#" + "ت" * 30, "AI", "LLM", "extra"],
    )
    with patch.object(summarizer, "_call_model", side_effect=[bad, bad]):
        s = summarizer.summarize(ITEM)
    assert len(s.title) <= TITLE_MAX and len(s.hook) <= HOOK_MAX
    assert all(len(b) <= BULLET_MAX for b in s.bullets)
    assert all("http" not in b and "🚀" not in b for b in s.bullets)
    assert len(s.hashtags) <= 3 and all(len(h) <= HASHTAG_MAX for h in s.hashtags)


def test_raises_when_bullets_unrecoverable():
    bad = model_json(bullets=["فقط یک مورد"])
    with patch.object(summarizer, "_call_model", side_effect=[bad, bad]):
        with pytest.raises(summarizer.SummarizationError):
            summarizer.summarize(ITEM)


def test_raises_when_never_valid_json():
    with patch.object(summarizer, "_call_model", side_effect=["nope", "still nope"]):
        with pytest.raises(summarizer.SummarizationError):
            summarizer.summarize(ITEM)


# ----------------------------- templates ----------------------------------- #
def max_summary():
    return Summary(
        title="ع" * TITLE_MAX,
        hook="ه" * HOOK_MAX,
        bullets=tuple("ب" * BULLET_MAX for _ in range(3)),
        hashtags=tuple("ت" * HASHTAG_MAX for _ in range(3)),
    )


def test_worst_case_caption_fits():
    brand = t.Branding(channel_handle="@" + "c" * 31)
    caption = t.build_caption(ITEM, max_summary(), brand)
    assert t.telegram_length(caption) <= 800
    assert "<i>" in caption  # hook kept when everything is within limits


def test_html_in_llm_text_is_escaped():
    s = Summary("A <script> & B", "x < y", ("1 > 0", "b", "c"), ("t",))
    caption = t.build_caption(ITEM, s, t.Branding())
    assert "<script>" not in caption and "&lt;script&gt;" in caption
    assert "&amp;" in caption


def test_oversized_summary_drops_hook_first():
    s = Summary("t", "هوک " * 30, tuple("ب" * 220 for _ in range(3)), ("a",))
    caption = t.build_caption(ITEM, s, t.Branding())
    assert "<i>" not in caption and t.telegram_length(caption) <= 800


def test_hopeless_summary_raises():
    s = Summary("t", "h", tuple("ب" * 400 for _ in range(3)), ())
    with pytest.raises(t.MessageTooLong):
        t.build_caption(ITEM, s, t.Branding())


def test_hashtag_formatting():
    out = t.format_hashtags(["هوش\u200cمصنوعی", "AI!", "ai", "  "], fixed=["اخبار_تکنولوژی"])
    assert out == "#هوش_مصنوعی #AI #اخبار_تکنولوژی"


def test_reply_markup():
    markup = t.build_reply_markup(ITEM, "https://telegra.ph/x")
    assert [b["url"] for b in markup["inline_keyboard"][0]] == ["https://telegra.ph/x", ITEM.url]
    assert t.build_reply_markup(NewsItem("i", "arxiv", "t", "ftp://bad"), None) is None


def test_digest_stays_under_limit_and_reports_count():
    entries = [(ITEM, max_summary(), "https://telegra.ph/p-%d" % i) for i in range(100)]
    text, included = t.build_digest(entries, "2026-10-31", t.Branding())
    assert t.telegram_length(text) <= 4096
    assert 0 < included < 100
    assert text.count("<a href=") == included
