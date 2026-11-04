"""
Command line entry point used by the GitHub Actions workflows (and for local runs).

    python -m src.main run            fetch, summarize, publish (mode "single") or keep for the digest
    python -m src.main digest         post the daily digest
    python -m src.main check          verify the Telegram token / channel / bot permissions (sends nothing)
    add --dry-run to run/digest       log what would be posted; sends nothing, publishes nothing, saves nothing

Exit codes: 0 ok (including "stopped early because of quota": that is normal on a free tier),
            2 invalid configuration, 3 corrupt state file, 4 Telegram configuration problem.
Anything else is an unexpected bug and surfaces as a normal traceback (exit 1).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence

from src.config import Config, ConfigError
from src.ingestion.arxiv import ArxivSource
from src.ingestion.base import Source, make_session
from src.ingestion.github_trending import GitHubTrendingSource
from src.ingestion.hackernews import HackerNewsSource
from src.llm.client import LLMConfigError
from src.pipeline import Pipeline, RunStats
from src.publishing.telegram import TelegramClient, TelegramError
from src.publishing.telegraph import TelegraphError
from src.storage.state import StateCorruptError, StateStore

logger = logging.getLogger("bot")


def build_sources(cfg: Config) -> list[Source]:
    session = make_session()
    factories = {
        "hackernews": lambda: HackerNewsSource(session, min_points=cfg.hn_min_points),
        "github": lambda: GitHubTrendingSource(session),
        "arxiv": lambda: ArxivSource(session, categories=cfg.arxiv_categories, keywords=cfg.arxiv_keywords),
    }
    return [factories[name]() for name in cfg.sources]


def check_secrets(command: str, cfg: Config, env=os.environ) -> None:
    """Fail BEFORE any request is made (and before any quota is spent) if a needed secret is missing.
    A forgotten repository secret otherwise only shows up deep inside the run."""
    needed: list[str] = []
    if command == "run" and not (env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY")):
        needed.append("GEMINI_API_KEY")  # needed even for dry runs: they call Gemini to show you the output
    if command == "run" and cfg.write_articles and not cfg.dry_run and not env.get("TELEGRAPH_ACCESS_TOKEN"):
        needed.append("TELEGRAPH_ACCESS_TOKEN")
    if not cfg.dry_run:
        needed += [name for name in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID") if not env.get(name)]
    if needed:
        raise ConfigError(
            f"missing secret(s): {', '.join(needed)}. Add them under Settings > Secrets and variables > Actions > Secrets"
        )


def _annotate(level: str, title: str, message: str) -> None:
    """GitHub Actions workflow command: shows up as a yellow/red banner on the run page."""
    print(f"::{level} title={title}::{message}")


def _write_job_summary(command: str, stats: RunStats, dry_run: bool) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"### `{command}`{' (dry run)' if dry_run else ''}\n\n`{stats.render()}`\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.main", description=__doc__.split("\n")[1].strip())
    parser.add_argument("command", choices=("run", "digest", "check"))
    parser.add_argument("--dry-run", action="store_true", help="log instead of sending; nothing is saved")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    try:
        cfg = Config.from_env()
        if args.dry_run:
            cfg = Config(**{**cfg.__dict__, "dry_run": True})

        if args.command == "check":
            info = TelegramClient.from_env().check()
            print(f"bot @{info['bot']} -> {info['chat_type']} '{info['chat']}' as {info['status']}")
            if not info["can_post"]:
                _annotate("error", "Telegram", "the bot is not allowed to post: make it an administrator with 'Post messages'")
                return 4
            print("OK: the bot can post to the channel")
            return 0

        check_secrets(args.command, cfg)
        store = StateStore(cfg.state_dir, persist=not cfg.dry_run)
        telegram = None if cfg.dry_run else TelegramClient.from_env()
        pipeline = Pipeline(store, telegram, cfg, sources=build_sources(cfg) if args.command == "run" else ())
        stats = pipeline.run() if args.command == "run" else pipeline.digest()
    except (ConfigError, LLMConfigError) as exc:
        _annotate("error", "Invalid configuration", str(exc))
        return 2
    except TelegraphError as exc:  # only invalid-token errors get this far; other Telegraph problems are handled per item
        _annotate("error", "Telegra.ph", str(exc))
        return 2
    except ValueError as exc:  # missing TELEGRAM_* variables raise ValueError from TelegramClient.from_env
        _annotate("error", "Invalid configuration", str(exc))
        return 2
    except StateCorruptError as exc:
        _annotate("error", "State file", f"{exc}. Fix or delete the `state` branch deliberately; the bot will not reset it.")
        return 3
    except TelegramError as exc:
        _annotate("error", "Telegram", str(exc))
        return 4

    print(f"{args.command}: {stats.render()}")
    if stats.stopped:
        _annotate("warning", "Stopped early", stats.stopped)
    _write_job_summary(args.command, stats, cfg.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
