"""
Telegram Bot API sender (sendMessage + a setup check), using plain `requests`.

Retry policy (this matters: Telegram has no idempotency keys, so a blind retry can post twice)
    retried     HTTP 429 with a short `retry_after`; failures where the request provably never reached
                Telegram (connect timeout, DNS failure, connection refused)
    NOT retried a read timeout, a dropped connection mid-request or an HTTP 5xx: the message may already
                be in the channel. These raise TelegramError(delivery_unknown=True) so the caller can
                choose "possibly duplicate" or "possibly lost" deliberately (the pipeline picks "lost":
                it marks the item FAILED rather than risk a double post).
    read-only   getMe/getChat/getChatMember are always safe to retry.

Secrets: `requests` puts the full URL (which contains the bot token) into its exception messages.
Every error raised here is scrubbed of the token and raised `from None`.

Limits: text is 1-4096 characters after entity parsing; Telegram throttles a channel to roughly
20 messages per minute, so consecutive sends are spaced `min_interval` seconds apart.
"""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import Any

import requests
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError

from src.publishing.templates import MESSAGE_LIMIT, telegram_length

API_BASE = "https://api.telegram.org"
MAX_RETRY_AFTER_SECONDS = 30  # longer flood waits are raised, not slept (CI minutes)

logger = logging.getLogger(__name__)


class TelegramError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        error_code: int | None = None,
        retry_after: int | None = None,
        delivery_unknown: bool = False,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.retry_after = retry_after
        self.delivery_unknown = delivery_unknown  # True: the message may or may not be in the channel

    @property
    def fatal(self) -> bool:
        """Configuration problems that no retry can fix (bad token, bot removed, wrong chat id).
        The run should fail loudly instead of quietly skipping every item."""
        return self.error_code in (401, 403, 404) or "chat not found" in str(self).lower()


def _never_sent(exc: requests.RequestException) -> bool:
    """True when the failure happened before any bytes of the request reached Telegram."""
    if isinstance(exc, requests.ConnectTimeout):
        return True
    if isinstance(exc, requests.ConnectionError) and exc.args:
        reason = getattr(exc.args[0], "reason", None)  # requests wraps urllib3's MaxRetryError
        return isinstance(reason, (NewConnectionError, ConnectTimeoutError))
    return False


class TelegramClient:
    def __init__(
        self,
        token: str,
        chat_id: str | int,
        *,
        session: requests.Session | None = None,
        connect_timeout: float = 5.0,
        read_timeout: float = 20.0,
        max_retries: int = 3,
        min_interval: float = 3.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not token or not chat_id:
            raise ValueError("a bot token and a chat id are required")
        self._token = token
        self._chat_id = chat_id
        self._session = session or requests.Session()
        self._timeout = (connect_timeout, read_timeout)
        self._max_retries = max_retries
        self._min_interval = min_interval
        self._sleep = sleep
        self._clock = clock
        self._last_send: float | None = None

    @classmethod
    def from_env(cls, **kwargs: Any) -> "TelegramClient":
        token, chat_id = os.environ.get("TELEGRAM_BOT_TOKEN", ""), os.environ.get("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set")
        return cls(token, chat_id, **kwargs)

    # ------------------------------------------------------------------ #
    def send_message(
        self,
        text: str,
        *,
        reply_markup: dict[str, Any] | None = None,
        preview_url: str | None = None,
        silent: bool = False,
    ) -> int:
        """Send an HTML message to the channel and return its message_id.

        Links are not previewed unless `preview_url` is given (e.g. the Telegraph page, which
        Telegram renders as an Instant View card).
        """
        if telegram_length(text) > MESSAGE_LIMIT:
            raise ValueError(f"message is longer than Telegram's {MESSAGE_LIMIT} character limit")
        self._throttle()
        payload: dict[str, Any] = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "HTML",
            "link_preview_options": {"url": preview_url} if preview_url else {"is_disabled": True},
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup
        if silent:
            payload["disable_notification"] = True
        try:
            result = self._call("sendMessage", payload)
        finally:
            self._last_send = self._clock()
        return int(result["message_id"])

    def check(self) -> dict[str, Any]:
        """Verify the token, the chat id and the bot's right to post (for channels). Sends nothing."""
        me = self._call("getMe", {}, read_only=True)
        chat = self._call("getChat", {"chat_id": self._chat_id}, read_only=True)
        member = self._call("getChatMember", {"chat_id": self._chat_id, "user_id": me["id"]}, read_only=True)
        status = member.get("status")
        return {
            "bot": me.get("username"),
            "chat": chat.get("title") or chat.get("username") or str(chat.get("id")),
            "chat_type": chat.get("type"),
            "status": status,
            "can_post": status == "creator" or bool(member.get("can_post_messages")),
        }

    # ------------------------------------------------------------------ #
    def _throttle(self) -> None:
        if self._last_send is None:
            return
        wait = self._min_interval - (self._clock() - self._last_send)
        if wait > 0:
            self._sleep(wait)

    def _redact(self, text: str) -> str:
        return text.replace(self._token, "<bot-token>")

    def _call(self, method: str, payload: dict[str, Any], *, read_only: bool = False) -> dict[str, Any]:
        url = f"{API_BASE}/bot{self._token}/{method}"
        for attempt in range(1, self._max_retries + 1):
            last = attempt == self._max_retries
            try:
                response = self._session.post(url, json=payload, timeout=self._timeout)
            except requests.RequestException as exc:
                safe_to_retry = read_only or _never_sent(exc)
                message = self._redact(f"{method} request failed: {type(exc).__name__}: {exc}")
                if safe_to_retry and not last:
                    logger.warning("%s (retrying, attempt %d)", message, attempt)
                    self._sleep(2 ** attempt)
                    continue
                raise TelegramError(message, delivery_unknown=not safe_to_retry) from None

            try:
                data = response.json()
            except ValueError:
                data = None
            if isinstance(data, dict) and data.get("ok"):
                return data["result"]

            body = data if isinstance(data, dict) else {}
            code = int(body.get("error_code") or response.status_code)
            description = body.get("description") or f"HTTP {response.status_code}"
            retry_after = (body.get("parameters") or {}).get("retry_after")

            if code == 429 and retry_after is not None and retry_after <= MAX_RETRY_AFTER_SECONDS and not last:
                logger.warning("Telegram flood control: waiting %ss", retry_after)
                self._sleep(retry_after + 1)
                continue
            if code >= 500 and read_only and not last:
                self._sleep(2 ** attempt)
                continue
            raise TelegramError(
                self._redact(f"{method}: {description}"),
                error_code=code,
                retry_after=retry_after,
                delivery_unknown=code >= 500 and not read_only,
            )
        raise TelegramError(f"{method}: retries exhausted")  # pragma: no cover
