"""
Runtime settings, read from environment variables (GitHub Actions `vars.*` / local .env).

GitHub quirk this module exists to handle: an unset `${{ vars.NAME }}` expands to an EMPTY STRING,
not to "missing". Every setting therefore treats "" as "use the default".
Secrets (GEMINI_API_KEY, TELEGRAM_*, TELEGRAPH_ACCESS_TOKEN) are read where they are used, not here.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

PUBLISH_MODES = ("single", "digest")
KNOWN_SOURCES = ("hackernews", "github", "arxiv")


class ConfigError(ValueError):
    """A setting has an invalid value. Raised early so the run fails before spending any quota."""


def _get(env: Mapping[str, str], name: str, default: str) -> str:
    value = (env.get(name) or "").strip()
    return value or default


def _int(env: Mapping[str, str], name: str, default: int, *, minimum: int = 0) -> int:
    raw = _get(env, name, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from None
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _get(env, name, "true" if default else "false").lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{name} must be true or false, got {raw!r}")


def _list(env: Mapping[str, str], name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = _get(env, name, "")
    return tuple(part.strip() for part in raw.split(",") if part.strip()) if raw else default


@dataclass(frozen=True)
class Config:
    publish_mode: str = "single"           # single: post each item as it is processed | digest: only summarize, post via `digest`
    items_per_source: int = 2              # NEW items processed per source per run (Gemini free-tier quota is the real limit)
    max_posts_per_run: int = 6             # hard cap on single posts per run (incl. leftovers from earlier runs)
    write_articles: bool = True            # also generate the full Persian article and publish it to Telegraph
    max_runtime_seconds: int = 900         # stop STARTING new items after this; the job's own timeout is the backstop
    digest_size: int = 10                  # max entries in one digest message
    digest_max_age_hours: int = 48         # older READY items are left out of the digest
    state_dir: str = ".state"
    sources: tuple[str, ...] = KNOWN_SOURCES
    hn_min_points: int = 100
    arxiv_categories: tuple[str, ...] = ("cs.AI", "cs.CL", "cs.LG")
    arxiv_keywords: tuple[str, ...] = ()   # empty = keep every paper (usually too many; set a few keywords)
    dry_run: bool = False                  # log instead of sending; nothing is saved, nothing is published

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Config":
        env = os.environ if env is None else env
        mode = _get(env, "PUBLISH_MODE", cls.publish_mode).lower()
        if mode not in PUBLISH_MODES:
            raise ConfigError(f"PUBLISH_MODE must be one of {PUBLISH_MODES}, got {mode!r}")
        sources = tuple(s.lower() for s in _list(env, "SOURCES", KNOWN_SOURCES))
        unknown = [s for s in sources if s not in KNOWN_SOURCES]
        if unknown:
            raise ConfigError(f"SOURCES contains unknown names {unknown}; choose from {KNOWN_SOURCES}")
        return cls(
            publish_mode=mode,
            items_per_source=_int(env, "ITEMS_PER_SOURCE", cls.items_per_source, minimum=1),
            max_posts_per_run=_int(env, "MAX_POSTS_PER_RUN", cls.max_posts_per_run, minimum=1),
            write_articles=_bool(env, "WRITE_ARTICLES", cls.write_articles),
            max_runtime_seconds=_int(env, "MAX_RUNTIME_SECONDS", cls.max_runtime_seconds, minimum=30),
            digest_size=_int(env, "DIGEST_SIZE", cls.digest_size, minimum=1),
            digest_max_age_hours=_int(env, "DIGEST_MAX_AGE_HOURS", cls.digest_max_age_hours, minimum=1),
            state_dir=_get(env, "STATE_DIR", cls.state_dir),
            sources=sources,
            hn_min_points=_int(env, "HN_MIN_POINTS", cls.hn_min_points),
            arxiv_categories=_list(env, "ARXIV_CATEGORIES", cls.arxiv_categories),
            arxiv_keywords=_list(env, "ARXIV_KEYWORDS", cls.arxiv_keywords),
            dry_run=_bool(env, "DRY_RUN", cls.dry_run),
        )
