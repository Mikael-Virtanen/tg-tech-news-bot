"""Run with: pytest -q   (no network or API key needed)."""
import json
from unittest.mock import MagicMock, patch

import pytest

from src.llm import article as art
from src.llm.prompts import ARTICLE_MAX_WORDS, ARTICLE_MIN_WORDS, normalize_article
from src.models import NewsItem
from src.publishing import telegraph as tg
from src.publishing import templates as t

ITEM = NewsItem(id="arxiv:2401.1", source="arxiv", title="Paper", url="https://arxiv.org/abs/2401.1", text="abstract")


def words(n: int) -> str:
    return " ".join(["کلمه"] * n)


def body_of(n: int = 150) -> str:
    return (
        f"{words(30)}\n\n## بخش اول\n{words(n // 2)}\n\n- مورد یک\n- مورد دو\n\n"
        f"## جمع‌بندی\n{words(n // 4)}"
    )


def article_json(title: str = "عنوان آزمایشی", body: str | None = None) -> str:
    return json.dumps({"title": title, "body": body if body is not None else body_of()}, ensure_ascii=False)


# ------------------------------ normalization ------------------------------ #
def test_normalize_strips_markup_the_model_was_told_not_to_use():
    raw = (
        "این **مهم** است `code` و https://x.com/a (https://y.com) <b>تگ</b> 🚀\n\n"
        "#### زیرعنوان\n```python\nprint(1)\n```\n*ایتالیک* متن"
    )
    out = normalize_article({"title": "ي عنوان كوتاه", "body": raw})
    body = out["body"]
    assert out["title"] == "ی عنوان کوتاه"
    for junk in ("**", "`", "http", "<b>", "🚀", "```", "()"):
        assert junk not in body
    assert "این مهم است code و" in body
    assert "### زیرعنوان" in body
    assert "ایتالیک متن" in body


# ------------------------------ write_article ------------------------------ #
def test_valid_first_attempt():
    with patch.object(art, "call_model", side_effect=[article_json()]) as call:
        result = art.write_article(ITEM)
    assert call.call_count == 1
    assert result.title == "عنوان آزمایشی"
    assert len(result.body.split()) >= ARTICLE_MIN_WORDS


def test_too_short_triggers_retry_with_feedback():
    with patch.object(art, "call_model", side_effect=[article_json(body=words(20)), article_json()]) as call:
        result = art.write_article(ITEM)
    assert call.call_count == 2
    assert "the minimum is 80" in call.call_args_list[1].args[0][-1]["content"]
    assert result.body.startswith("کلمه")


def test_overlong_body_is_trimmed_to_whole_blocks_without_orphan_heading():
    long_body = f"{words(880)}\n\n## عنوان یتیم\n\n{words(100)}"
    payload = article_json(body=long_body)
    with patch.object(art, "call_model", side_effect=[payload, payload]):
        result = art.write_article(ITEM)
    assert len(result.body.split()) <= ARTICLE_MAX_WORDS
    assert "یتیم" not in result.body  # heading whose text was cut is dropped too


def test_overlong_title_is_shortened_by_fallback():
    payload = article_json(title="ع " * 80)
    with patch.object(art, "call_model", side_effect=[payload, payload]):
        result = art.write_article(ITEM)
    assert len(result.title) <= 100


def test_raises_when_still_too_short_after_retry():
    payload = article_json(body=words(10))
    with patch.object(art, "call_model", side_effect=[payload, payload]):
        with pytest.raises(art.ArticleError):
            art.write_article(ITEM)


def test_raises_when_never_valid_json():
    with patch.object(art, "call_model", side_effect=["{broken", "nope"]):
        with pytest.raises(art.ArticleError):
            art.write_article(ITEM)


# --------------------------- end to end (mocked) --------------------------- #
def test_article_to_telegraph_to_keyboard():
    big = article_json(body=body_of(800))  # near the word cap
    with patch.object(art, "call_model", side_effect=[big]):
        article = art.write_article(ITEM)

    session = MagicMock()
    session.post.return_value.json.return_value = {
        "ok": True,
        "result": {"url": "https://telegra.ph/demo-10-31", "path": "demo-10-31", "title": article.title},
    }
    page = tg.TelegraphClient("tok", session=session).publish(article.title, article.body, source_url=ITEM.url)

    content = session.post.call_args.kwargs["data"]["content"]
    nodes = json.loads(content)
    assert {"p", "h3", "ul"} <= {n["tag"] for n in nodes if isinstance(n, dict)}
    assert len(content.encode("utf-8")) < tg.MAX_CONTENT_BYTES

    markup = t.build_reply_markup(ITEM, page.url)
    assert markup["inline_keyboard"][0][0]["url"] == "https://telegra.ph/demo-10-31"
