# Tech & AI news → Telegram

Fetches Hacker News, GitHub Trending and arXiv, writes Persian summaries and full articles with the
Gemini API, publishes the articles to Telegra.ph (Instant View) and posts to your Telegram channel.
Runs on **GitHub Actions** on a schedule. No server, no database, no credit card.

```
cron ─► restore state ─► fetch (skip known ids) ─► pick best N per source ─► Gemini summary
        (state branch)                                                        └► Gemini article ─► Telegra.ph
                                                  post / digest ◄─────────────────────────────────┘
        save state ◄─ (always, even after a crash)
```

## Setup (about 15 minutes)

1. **Repository.** Create a GitHub repo and push this project. *Public* repos get unlimited Actions minutes on
   standard runners; *private* repos get 2,000 minutes/month, which is plenty for this bot (estimate: a few minutes per run).
2. **Telegram.** Talk to [@BotFather](https://t.me/BotFather) → `/newbot` → copy the token. Create a channel, then add the
   bot as an **administrator with the "Post messages" right**. The chat id is `@your_channel_username` for a public channel.
3. **Gemini.** Create a free key at <https://aistudio.google.com/apikey>.
4. **Telegra.ph.** Once, locally: `pip install -r requirements.txt` then
   `python -m src.publishing.telegraph "ShortName" "Channel name" "https://t.me/your_channel"`. It prints an access token.
5. **Secrets** (Settings → Secrets and variables → Actions → *Secrets*): `GEMINI_API_KEY`, `TELEGRAM_BOT_TOKEN`,
   `TELEGRAM_CHAT_ID`, `TELEGRAPH_ACCESS_TOKEN`.
   **Variables** (same page, *Variables*, all optional): `CHANNEL_HANDLE` (e.g. `@your_channel`, shown in every footer),
   `PUBLISH_MODE`, `ARXIV_KEYWORDS`, … see `.env.example` for the full list.
6. **Check the bot can post:** locally, with the two Telegram values exported, run `python -m src.main check`.
7. **First run:** Actions tab → *Tech news bot* → *Run workflow*. The default is a **dry run**: it calls Gemini and prints
   the posts and article excerpts it *would* send, but sends, publishes and saves nothing. Read the Persian output.
   Then run again with *dry run* unticked. After that the schedule takes over.

## Modes

| `PUBLISH_MODE` | Behaviour |
|---|---|
| `single` (default) | Every processed item is posted right away, up to `MAX_POSTS_PER_RUN` per run. Nothing to configure: the workflows pin it. `digest.yml` is skipped automatically in this mode. |
| `digest` | `bot.yml` only summarizes; `digest.yml` (daily) posts **one** message with the best items. |

## How the state works

`.state/news.json` lives on a dedicated **`state` branch**, restored at the start of every run and force-pushed as a
single commit at the end, so it never accumulates history. It stores ids, statuses, summaries and Telegraph links, never
article text. Records expire after 30 days (which is also how long an item stays "already posted").

* A crash or timeout still saves progress (the save step uses `if: always()`), otherwise the next run would re-post.
* If Telegram might or might not have delivered a post (timeout, 5xx), the item is marked FAILED instead of retried:
  a duplicate in a news channel is worse than a missing item. Check the channel if you see that warning.
* If the `state` branch can't be reached, the run stops; it never "starts fresh" on a network error.
* A corrupt state file stops the run (exit code 3). Fix it by hand; the bot never resets it silently.

## Things to know about the free setup

* **Gemini quota.** The free tier has per-minute and per-day limits (see your project's page in AI Studio). When it is
  exhausted (or Gemini is down) the run stops with a yellow warning; unprocessed items are simply retried next run. Lower `ITEMS_PER_SOURCE` if it happens often.
  On the free tier Google may use prompts to improve its products: only public content is sent.
* **Schedules are best-effort.** GitHub may delay scheduled runs by many minutes and gives no guarantee. The cron minutes are
  deliberately off-peak (`:17`, `:43`).
* **60-day rule.** In public repositories GitHub disables scheduled workflows after 60 days without repository activity.
  `scripts/keepalive.sh` adds an empty commit to the default branch when it is 45+ days old. If the default branch is protected
  that step only warns; then check the Actions tab now and then.
* **Dry runs** never touch Telegram, Telegra.ph or the state, so they are safe to repeat.

## Local use

```bash
pip install -r requirements-dev.txt
python -m pytest -q
export GEMINI_API_KEY=...            # and the other values from .env.example
python -m src.main run --dry-run
```
