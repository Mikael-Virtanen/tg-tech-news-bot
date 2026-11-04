from unittest.mock import MagicMock

import pytest
import requests
from urllib3.exceptions import MaxRetryError, NewConnectionError

from src.publishing.telegram import TelegramClient, TelegramError

TOKEN = "123456:SECRET-token-value"


def ok(result):
    response = MagicMock(status_code=200)
    response.json.return_value = {"ok": True, "result": result}
    return response


def fail(code, description, retry_after=None):
    response = MagicMock(status_code=code)
    body = {"ok": False, "error_code": code, "description": description}
    if retry_after is not None:
        body["parameters"] = {"retry_after": retry_after}
    response.json.return_value = body
    return response


def make(*responses, **kwargs):
    session = MagicMock()
    session.post.side_effect = list(responses)
    sleeps = []
    client = TelegramClient(TOKEN, "@channel", session=session, sleep=sleeps.append, **kwargs)
    return client, session, sleeps


# ------------------------------- request shape ------------------------------ #
def test_send_message_payload_and_message_id():
    client, session, _ = make(ok({"message_id": 4242}))
    markup = {"inline_keyboard": [[{"text": "x", "url": "https://t.me"}]]}
    assert client.send_message("<b>hi</b>", reply_markup=markup) == 4242

    args, kwargs = session.post.call_args
    assert args[0] == f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    assert kwargs["json"] == {
        "chat_id": "@channel", "text": "<b>hi</b>", "parse_mode": "HTML",
        "link_preview_options": {"is_disabled": True}, "reply_markup": markup,
    }
    assert kwargs["timeout"] == (5.0, 20.0)


def test_preview_url_and_silent_flags():
    client, session, _ = make(ok({"message_id": 1}))
    client.send_message("x", preview_url="https://telegra.ph/a", silent=True)
    payload = session.post.call_args.kwargs["json"]
    assert payload["link_preview_options"] == {"url": "https://telegra.ph/a"}
    assert payload["disable_notification"] is True


def test_overlong_message_is_rejected_before_any_request():
    client, session, _ = make()
    with pytest.raises(ValueError):
        client.send_message("x" * 4097)
    session.post.assert_not_called()


# ------------------------------- retry policy ------------------------------- #
def test_flood_control_is_waited_out_then_retried():
    client, session, sleeps = make(fail(429, "Too Many Requests: retry after 2", retry_after=2), ok({"message_id": 9}))
    assert client.send_message("x") == 9
    assert sleeps == [3] and session.post.call_count == 2


def test_long_flood_wait_is_raised_not_slept():
    client, session, sleeps = make(fail(429, "Too Many Requests", retry_after=120))
    with pytest.raises(TelegramError) as info:
        client.send_message("x")
    assert info.value.error_code == 429 and info.value.retry_after == 120
    assert not info.value.delivery_unknown and not info.value.fatal  # definitely not delivered
    assert sleeps == []


def test_connect_failures_are_retried_because_nothing_was_sent():
    refused = requests.ConnectionError(MaxRetryError(None, "u", reason=NewConnectionError(None, "refused")))
    client, session, sleeps = make(requests.ConnectTimeout("slow"), refused, ok({"message_id": 3}))
    assert client.send_message("x") == 3
    assert session.post.call_count == 3 and sleeps == [2, 4]


def test_read_timeout_is_never_retried_because_the_post_may_exist():
    client, session, _ = make(requests.ReadTimeout("no reply"), ok({"message_id": 1}))
    with pytest.raises(TelegramError) as info:
        client.send_message("x")
    assert info.value.delivery_unknown is True
    assert session.post.call_count == 1  # a retry here could publish the same post twice


def test_server_errors_are_flagged_delivery_unknown_and_not_retried():
    client, session, _ = make(fail(502, "Bad Gateway"))
    with pytest.raises(TelegramError) as info:
        client.send_message("x")
    assert info.value.delivery_unknown is True and session.post.call_count == 1


def test_client_errors_are_definite_failures():
    client, _, _ = make(fail(400, "Bad Request: can't parse entities: Unsupported start tag"))
    with pytest.raises(TelegramError) as info:
        client.send_message("<oops>")
    assert info.value.error_code == 400 and not info.value.delivery_unknown and not info.value.fatal


@pytest.mark.parametrize(
    "code, description",
    [(401, "Unauthorized"), (403, "Forbidden: bot is not a member of the channel chat"),
     (404, "Not Found"), (400, "Bad Request: chat not found")],
)
def test_configuration_errors_are_fatal(code, description):
    client, _, _ = make(fail(code, description))
    with pytest.raises(TelegramError) as info:
        client.send_message("x")
    assert info.value.fatal


# ------------------------------- secrets ------------------------------------ #
def test_bot_token_never_appears_in_errors():
    leaky = requests.ReadTimeout(f"HTTPSConnectionPool: Read timed out. url=/bot{TOKEN}/sendMessage")
    client, _, _ = make(leaky)
    with pytest.raises(TelegramError) as info:
        client.send_message("x")
    assert TOKEN not in str(info.value) and TOKEN not in repr(info.value)
    assert info.value.__cause__ is None and info.value.__suppress_context__  # `from None`: no chained traceback

    client, _, _ = make(fail(400, f"weird description containing {TOKEN}"))
    with pytest.raises(TelegramError) as info:
        client.send_message("x")
    assert TOKEN not in str(info.value)


# ------------------------------- throttling --------------------------------- #
def test_consecutive_sends_are_spaced_out():
    now = [100.0]
    session = MagicMock()
    session.post.side_effect = [ok({"message_id": 1}), ok({"message_id": 2}), ok({"message_id": 3})]
    sleeps = []
    client = TelegramClient(TOKEN, "@c", session=session, sleep=sleeps.append, clock=lambda: now[0], min_interval=3.0)

    client.send_message("a")
    assert sleeps == []            # first send: no wait
    now[0] += 1.0
    client.send_message("b")
    assert sleeps == [2.0]         # 1 s elapsed of the 3 s interval
    now[0] += 10.0
    client.send_message("c")
    assert sleeps == [2.0]         # already far enough apart


# ------------------------------- setup check -------------------------------- #
def test_check_reports_bot_chat_and_posting_rights_and_retries_reads():
    client, session, _ = make(
        fail(502, "Bad Gateway"), ok({"id": 7, "username": "newsbot"}),
        ok({"id": -100, "title": "Tech News", "type": "channel"}),
        ok({"status": "administrator", "can_post_messages": True}),
    )
    assert client.check() == {"bot": "newsbot", "chat": "Tech News", "chat_type": "channel",
                              "status": "administrator", "can_post": True}
    assert session.post.call_args.kwargs["json"] == {"chat_id": "@channel", "user_id": 7}


def test_check_detects_a_bot_that_is_not_allowed_to_post():
    client, _, _ = make(ok({"id": 7, "username": "b"}), ok({"title": "T", "type": "channel"}),
                        ok({"status": "member"}))
    assert client.check()["can_post"] is False


def test_from_env_requires_both_values(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "")
    with pytest.raises(ValueError, match="TELEGRAM_CHAT_ID"):
        TelegramClient.from_env()
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "@c")
    assert isinstance(TelegramClient.from_env(), TelegramClient)
