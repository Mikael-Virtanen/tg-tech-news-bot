"""
Full-article generation for Telegra.ph: NewsItem -> validated Persian Article.

    call model -> parse JSON -> normalize (strip stray markup) -> validate
        |-- valid                      -> return
        |-- violations (attempt 1)     -> retry once with the exact problem list
        `-- violations (attempt 2)     -> repair only what is safely repairable, else raise

Unlike the 800-character summary, an article can't be repaired by cutting words mid-sentence.
The only fallbacks are (a) shortening an over-long title and (b) dropping whole trailing
blocks of an over-long body. A too-short or empty article raises ArticleError.

Operational notes:
    - A 600-word Persian article can take 30-60 s to generate, so run this in its own
      Lambda (or behind a queue) with a timeout of several minutes, not inside the same
      invocation that must answer a webhook.
    - Telegraph publishing (publishing/telegraph.py) is a separate step; this module does no I/O
      besides the OpenAI call.
"""
from __future__ import annotations

import logging
import re

from src.llm.client import LLMError, Messages, call_model, parse_json_object
from src.llm.prompts import (
    ARTICLE_MAX_WORDS,
    ARTICLE_RESPONSE_FORMAT,
    ARTICLE_SOURCE_CHARS,
    ARTICLE_SYSTEM_PROMPT,
    ARTICLE_TITLE_MAX,
    build_user_message,
    normalize_article,
    validate_article,
)
from src.models import Article, NewsItem

MAX_ATTEMPTS = 2  # 1 initial call + 1 corrective retry
MAX_COMPLETION_TOKENS = 6000  # Persian is token-hungry: ~900 words can reach 4-5k tokens
REQUEST_TIMEOUT = 120.0
TEMPERATURE = 0.4

logger = logging.getLogger(__name__)


class ArticleError(LLMError):
    """No usable article could be produced for this item."""


def _retry_message(problems: list[str]) -> str:
    listed = "\n".join(f"- {p}" for p in problems)
    return (
        "Your previous answer broke these rules:\n"
        f"{listed}\n\n"
        "Return the corrected JSON object only. Keep the same facts and structure. Do not invent "
        "information to reach a length, and do not drop essential information to shorten."
    )


# --------------------------------------------------------------------------- #
# Repairs (only the ones that can't damage the text)
# --------------------------------------------------------------------------- #
def _fit_title(title: str) -> str:
    if len(title) <= ARTICLE_TITLE_MAX:
        return title
    cut = title[: ARTICLE_TITLE_MAX - 1]
    space = cut.rfind(" ")
    if space >= ARTICLE_TITLE_MAX * 0.6:
        cut = cut[:space]
    return cut.rstrip(" ،,.:;-\u200c") + "…"


def _trim_to_max_words(body: str, max_words: int) -> str:
    """Keep whole blocks (paragraph / heading / bullet list) until the word cap is reached.
    A heading or divider left dangling at the end is removed."""
    kept: list[str] = []
    total = 0
    for block in re.split(r"\n{2,}", body):
        words = len(block.split())
        if total + words > max_words:
            break
        kept.append(block)
        total += words
    while kept and (kept[-1].lstrip().startswith("#") or kept[-1].strip() == "---"):
        kept.pop()
    return "\n\n".join(kept)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def write_article(item: NewsItem) -> Article:
    """Produce a validated Persian article for `item`, or raise ArticleError / LLMError."""
    messages: Messages = [
        {"role": "system", "content": ARTICLE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_message(
                item.title, item.source, item.url, item.text, max_text_chars=ARTICLE_SOURCE_CHARS
            ),
        },
    ]
    latest: dict[str, str] | None = None  # most recent parseable output, kept for the fallback

    for attempt in range(1, MAX_ATTEMPTS + 1):
        raw = call_model(
            messages,
            ARTICLE_RESPONSE_FORMAT,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            timeout=REQUEST_TIMEOUT,
            temperature=TEMPERATURE,
        )
        parsed, problems = parse_json_object(raw)
        data: dict[str, str] | None = None
        if parsed is not None:
            data = normalize_article(parsed)
            latest = data
            problems = validate_article(data)
        if not problems:
            logger.info("Article for %s accepted on attempt %d", item.id, attempt)
            return Article(title=data["title"], body=data["body"])  # type: ignore[index]

        logger.warning("Article for %s failed validation (attempt %d): %s", item.id, attempt, problems)
        if attempt < MAX_ATTEMPTS:
            messages = [
                *messages,
                {"role": "assistant", "content": raw},
                {"role": "user", "content": _retry_message(problems)},
            ]

    if latest is None:
        raise ArticleError(f"model never returned valid JSON for {item.id}")

    repaired = {
        "title": _fit_title(latest["title"]),
        "body": _trim_to_max_words(latest["body"], ARTICLE_MAX_WORDS),
    }
    remaining = validate_article(repaired)
    if remaining:
        raise ArticleError(f"could not repair article for {item.id}: {remaining}")
    logger.warning("Article for %s was repaired by the fallback", item.id)
    return Article(title=repaired["title"], body=repaired["body"])
