"""Ingestion tests. No network: HTTP is faked and DNS is patched."""
import json
import socket
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.ingestion import base
from src.ingestion.arxiv import ArxivSource, arxiv_id, parse_feed
from src.ingestion.github_trending import GitHubTrendingSource, clean_readme, parse_trending
from src.ingestion.hackernews import HackerNewsSource
from src.models import NewsItem

FIXTURES = Path(__file__).parent / "fixtures"
REAL_GETADDRINFO = socket.getaddrinfo


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def fake_dns(host, port, *args, **kwargs):
    """Public IP for *.example hosts; real resolution (works offline) for IP literals like 127.0.0.1."""
    if host.endswith(".example"):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]
    return REAL_GETADDRINFO(host, port, *args, **kwargs)


@pytest.fixture(autouse=True)
def patched_dns():
    with patch.object(base.socket, "getaddrinfo", side_effect=fake_dns):
        yield


class FakeResponse:
    """Minimal stand-in for requests.Response, usable as a context manager like stream=True responses."""

    def __init__(self, content=b"", status=200, headers=None, json_data=None):
        self.content = content if isinstance(content, bytes) else content.encode()
        self.text = self.content.decode("utf-8", "replace")
        self.status_code = status
        self.headers = headers if headers is not None else {"Content-Type": "text/html; charset=utf-8"}
        self._json = json_data

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i : i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def session_with(routes):
    """requests.Session stand-in: `routes` maps a URL prefix (longest match wins) to a FakeResponse."""
    session = MagicMock()

    def get(url, **kwargs):
        for prefix in sorted(routes, key=len, reverse=True):
            if url.startswith(prefix):
                result = routes[prefix]
                if isinstance(result, Exception):
                    raise result
                return result
        raise AssertionError(f"unexpected request: {url}")

    session.get.side_effect = get
    return session


# --------------------------------- base ------------------------------------ #
@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://good.example/x",
        "http://127.0.0.1/",
        "http://localhost/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/admin",
        "http://[::1]/",
        "https://good.example:8443/",  # unusual port
        "not a url",
    ],
)
def test_ssrf_guard_rejects_dangerous_urls(url):
    assert not base.is_public_http_url(url)


def test_ssrf_guard_accepts_public_hosts():
    assert base.is_public_http_url("https://good.example/article")


def test_extract_main_text_keeps_content_and_drops_boilerplate():
    text = base.extract_main_text(fixture("article.html"))
    assert "log-structured design" in text
    assert "compaction scheduler adapts" in text
    assert "Engineers describe a new database engine" in text  # meta description is added
    for junk in ("newsletter", "Copyright 2026", "track = 1", "Short crumb"):
        assert junk not in text


def test_extract_main_text_respects_the_character_budget():
    html = "<article>" + "".join(f"<p>{'word ' * 30}{i}</p>" for i in range(200)) + "</article>"
    assert len(base.extract_main_text(html, max_chars=1000)) <= 1000


def test_fetch_page_returns_bytes_and_skips_non_html():
    session = session_with({"https://good.example/a": FakeResponse(b"<html>hi</html>")})
    assert base.fetch_page(session, "https://good.example/a") == b"<html>hi</html>"
    session = session_with({"https://good.example/p": FakeResponse(b"%PDF", headers={"Content-Type": "application/pdf"})})
    assert base.fetch_page(session, "https://good.example/p") is None


def test_fetch_page_revalidates_every_redirect_hop():
    redirect = FakeResponse(status=302, headers={"Location": "http://127.0.0.1:80/secret"})
    session = session_with({"https://good.example/r": redirect})
    assert base.fetch_page(session, "https://good.example/r") is None
    assert session.get.call_count == 1  # the internal address was never requested


def test_fetch_page_follows_safe_redirects():
    session = session_with({
        "https://good.example/old": FakeResponse(status=301, headers={"Location": "/new"}),
        "https://good.example/new": FakeResponse(b"<html>moved</html>"),
    })
    assert base.fetch_page(session, "https://good.example/old") == b"<html>moved</html>"


def test_fetch_page_truncates_oversized_bodies():
    session = session_with({"https://good.example/big": FakeResponse(b"x" * 100_000)})
    assert len(base.fetch_page(session, "https://good.example/big", max_bytes=20_000)) <= 20_000


def test_collect_items_isolates_failing_sources():
    good, bad = MagicMock(), MagicMock()
    good.name, bad.name = "good", "bad"
    good.fetch.return_value = [NewsItem("a:1", "a", "t", "https://x", "text")]
    bad.fetch.side_effect = RuntimeError("boom")
    assert [i.id for i in base.collect_items([bad, good])] == ["a:1"]


# ------------------------------- hacker news -------------------------------- #
def hn_session():
    return session_with({
        "https://hn.algolia.com/api/v1/search": FakeResponse(json_data=json.loads(fixture("hn_front_page.json"))),
    })


def test_hackernews_ranks_filters_and_falls_back_to_story_text():
    pages = {"https://blog.example.com/fast-db": fixture("article.html")}
    with patch("src.ingestion.hackernews.fetch_page", side_effect=lambda s, url: pages.get(url)):
        items = HackerNewsSource(hn_session(), min_points=100).fetch(limit=10)

    assert [i.id for i in items] == ["hackernews:1001", "hackernews:1002"]  # 1003 too few points, 1004 no text
    article, ask = items
    assert article.source == "hackernews" and article.score == 512
    assert article.url == "https://blog.example.com/fast-db"
    assert "log-structured design" in article.text
    assert ask.url == "https://news.ycombinator.com/item?id=1002"  # Ask HN has no link: discussion page
    assert "quantization settings" in ask.text and "<p>" not in ask.text


def test_hackernews_respects_limit_and_reports_api_failure():
    with patch("src.ingestion.hackernews.fetch_page", return_value=fixture("article.html")):
        assert len(HackerNewsSource(hn_session(), min_points=100).fetch(limit=1)) == 1
    broken = session_with({"https://hn.algolia.com": FakeResponse(status=503)})
    with pytest.raises(base.SourceError):
        HackerNewsSource(broken).fetch()


# ------------------------------ github trending ----------------------------- #
def test_parse_trending_extracts_all_fields_from_real_markup():
    repos = parse_trending(fixture("github_trending.html"))
    assert len(repos) == 3
    first = repos[0]
    assert first.full_name.count("/") == 1 and not first.full_name.startswith("/")
    assert first.description and first.language
    assert first.stars_today > 0 and first.total_stars >= first.stars_today


def test_clean_readme_removes_badges_images_html_and_link_targets():
    md = "# Proj\n[![CI](https://ci/badge.svg)](https://ci)\n![logo](logo.png)\n<p align='center'>Hi</p>\nSee [the docs](https://docs.example) now.\n\n\n\nEnd"
    out = clean_readme(md)
    assert "badge" not in out and "logo" not in out and "<p" not in out and "https://docs" not in out
    assert "See the docs now." in out and "\n\n\n" not in out


def github_session(trending=None, readme="# Tool\n" + "It does useful things for developers. " * 10):
    return session_with({
        "https://github.com/trending": trending if trending is not None else FakeResponse(fixture("github_trending.html")),
        "https://raw.githubusercontent.com/": FakeResponse(readme),
        "https://api.github.com/repos/": FakeResponse(readme),
        "https://api.github.com/search/repositories": FakeResponse(json_data={"items": [
            {"full_name": "new/repo", "description": "A brand new thing", "language": "Rust", "stargazers_count": 900},
        ]}),
    })


def test_github_source_builds_items_with_readme_text():
    items = GitHubTrendingSource(github_session(), token="t").fetch(limit=2)
    assert len(items) == 2
    first = items[0]
    assert first.id.startswith("github:") and first.id == first.id.lower()
    assert first.url.lower() == f"https://github.com/{first.id.split(':', 1)[1]}"
    assert "useful things" in first.text and first.score > 0
    assert items[0].score >= items[1].score


def requested_urls(session):
    return [call.args[0] for call in session.get.call_args_list]


def test_readme_comes_from_the_raw_host_and_tries_alternate_names():
    session = session_with({
        "https://raw.githubusercontent.com/": FakeResponse(status=404),
        "https://raw.githubusercontent.com/a/b/HEAD/readme.md": FakeResponse("# B\nA lowercase readme that works."),
    })
    assert "lowercase readme" in GitHubTrendingSource(session)._readme("a/b")
    assert requested_urls(session) == [
        "https://raw.githubusercontent.com/a/b/HEAD/README.md",
        "https://raw.githubusercontent.com/a/b/HEAD/readme.md",
    ]  # the rate-limited API was never touched


def test_readme_falls_back_to_the_api_with_token_when_raw_is_throttled():
    session = session_with({
        "https://raw.githubusercontent.com/": FakeResponse(status=429),
        "https://api.github.com/repos/a/b/readme": FakeResponse("API copy of the readme."),
    })
    assert GitHubTrendingSource(session, token="tok")._readme("a/b") == "API copy of the readme."
    assert requested_urls(session)[-1] == "https://api.github.com/repos/a/b/readme"
    assert session.get.call_args.kwargs["headers"]["Authorization"] == "Bearer tok"
    assert len(requested_urls(session)) == 2  # throttled raw host is tried once, not once per file name


def test_readme_is_empty_when_every_route_fails():
    session = session_with({
        "https://raw.githubusercontent.com/": FakeResponse(status=404),
        "https://api.github.com/": FakeResponse(status=403),
    })
    assert GitHubTrendingSource(session)._readme("a/b") == ""


def test_github_source_skips_repos_without_enough_text():
    assert GitHubTrendingSource(github_session(readme=""), min_text_chars=5000).fetch(limit=3) == []


def test_github_source_falls_back_to_search_api_when_markup_changes():
    items = GitHubTrendingSource(github_session(trending=FakeResponse(b"<html><body>redesigned</body></html>"))).fetch(limit=3)
    assert [i.id for i in items] == ["github:new/repo"]
    assert items[0].score == 900


def test_github_source_raises_when_everything_fails():
    session = session_with({
        "https://github.com/trending": FakeResponse(status=500),
        "https://api.github.com/search": FakeResponse(status=403),
    })
    with pytest.raises(base.SourceError):
        GitHubTrendingSource(session).fetch()


# ---------------------------------- arxiv ----------------------------------- #
def test_arxiv_id_drops_version():
    assert arxiv_id("http://arxiv.org/abs/2610.00001v3") == "2610.00001"
    assert arxiv_id("http://arxiv.org/abs/hep-th/9901001v1") == "hep-th/9901001"


def test_parse_feed_normalizes_entries_and_drops_thin_abstracts():
    items = parse_feed(fixture("arxiv_feed.xml"))
    assert [i.id for i in items] == ["arxiv:2610.00001", "arxiv:2610.00002"]  # "Tiny abstract" dropped
    first = items[0]
    assert first.title == "Long-Horizon Agent Planning with Memory"  # newlines collapsed
    assert first.url == "https://arxiv.org/abs/2610.00001v2"
    assert first.text.startswith("Authors: Ada Lovelace, Alan Turing\nCategory: cs.AI\nAbstract: We study how language model")


def test_parse_feed_keyword_filter_and_ranking():
    items = parse_feed(fixture("arxiv_feed.xml"), keywords=("agent", "planning"))
    assert [i.id for i in items] == ["arxiv:2610.00001"]  # the kernels paper matches neither
    assert items[0].score == 2  # "agent" matches "agents" (word-start), "planning" matches the title


def test_parse_feed_keywords_match_word_starts_not_substrings():
    feed = fixture("arxiv_feed.xml").replace(b"sequence lengths", b"the real world")  # abstract now contains "wo-rl-d"
    assert parse_feed(feed, keywords=("rl",)) == []  # substring matching would have kept the kernels paper
    assert parse_feed(fixture("arxiv_feed.xml"), keywords=("   ",)) != []  # blank keywords are ignored, not "match nothing"


def test_arxiv_source_builds_category_query_and_limits_results():
    session = session_with({"https://export.arxiv.org/api/query": FakeResponse(fixture("arxiv_feed.xml"))})
    items = ArxivSource(session, categories=("cs.AI", "cs.LG")).fetch(limit=1)
    assert len(items) == 1
    params = session.get.call_args.kwargs["params"]
    assert params["search_query"] == "cat:cs.AI OR cat:cs.LG"
    assert params["sortBy"] == "submittedDate" and params["max_results"] == 2


def test_arxiv_source_reports_api_failure():
    session = session_with({"https://export.arxiv.org": requests.ConnectionError("down")})
    with pytest.raises(base.SourceError):
        ArxivSource(session).fetch()


# ------------------------------- skip_ids ------------------------------------ #
def test_known_ids_are_skipped_before_any_download():
    """Known items must cost zero HTTP requests: that is the whole point of skip_ids."""
    # Hacker News: no article download for the skipped id
    fetched = []
    pages = {"https://blog.example.com/fast-db": fixture("article.html")}  # the PDF link yields nothing, as in reality
    with patch("src.ingestion.hackernews.fetch_page", side_effect=lambda s, url: fetched.append(url) or pages.get(url)):
        items = HackerNewsSource(hn_session(), min_points=100).fetch(10, skip_ids={"hackernews:1001"})
    assert [i.id for i in items] == ["hackernews:1002"] and "https://blog.example.com/fast-db" not in fetched

    # GitHub: no README request for the skipped repo
    session = github_session()
    first = GitHubTrendingSource(session).fetch(limit=1)[0]
    session = github_session()
    items = GitHubTrendingSource(session).fetch(limit=5, skip_ids={first.id})
    assert first.id not in [i.id for i in items]
    assert not any(first.id.split(":", 1)[1] in url for url in requested_urls(session) if "raw.githubusercontent" in url)

    # arXiv: entry dropped regardless of version suffix
    assert [i.id for i in parse_feed(fixture("arxiv_feed.xml"), skip_ids={"arxiv:2610.00001"})] == ["arxiv:2610.00002"]
