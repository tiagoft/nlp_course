"""Search Reddit without an API key. One file, Python 3.9+, standard library.

    python scraper.py neymar Barca psg          # try it from the shell

    posts = search_subreddit("Barca", "neymar")          # raises on failure
    results, errors = search_subreddits("neymar", subs)  # reports failures

Reddit's unauthenticated .json endpoints answer 403, so this reads the RSS
feeds. Two quirks it has to live with, both measured: repeated URLs get
rate-limited (the plain search URL returned 200 once in six tries, and six
times out of six once a throwaway feed/user pair was added), and bot user
agents are blocked. A 200 is also sometimes an empty feed, so empty results
are retried.
"""

from __future__ import annotations

import gzip
import re
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

__all__ = [
    "RedditPost",
    "SearchResult",
    "RedditClient",
    "RedditError",
    "search_subreddit",
    "search_subreddits",
    "health_check",
]

SEARCH_URL = "https://www.reddit.com/r/{subreddit}/search.rss"
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
ATOM = {"atom": "http://www.w3.org/2005/Atom"}
SORTS = ("relevance", "hot", "top", "new", "comments")
WINDOWS = ("all", "hour", "day", "week", "month", "year")


class RedditError(RuntimeError):
    """Reddit could not be reached, or answered with an error."""


@dataclass
class RedditPost:
    id: str
    subreddit: str
    title: str
    author: str
    url: str
    created_utc: datetime | None = None

    @property
    def recency(self) -> str:
        """Human-readable age, e.g. '3 days ago'."""
        if self.created_utc is None:
            return "unknown"
        seconds = (datetime.now(timezone.utc) - self.created_utc).total_seconds()
        for size, unit in ((31536000, "year"), (2592000, "month"), (86400, "day"),
                           (3600, "hour"), (60, "minute")):
            if seconds >= size:
                value = int(seconds // size)
                return f"{value} {unit}{'s' if value != 1 else ''} ago"
        return "just now"

    def to_dict(self) -> dict:
        data = self.__dict__.copy()
        data["created_utc"] = self.created_utc.isoformat() if self.created_utc else None
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "RedditPost":
        return cls(**{**data, "created_utc": _parse_time(data.get("created_utc"))})


@dataclass
class SearchResult:
    """One search, and how it went — including the ways it half-worked.

    ``search()`` collapses three outcomes into two: it raises, or it returns a
    list that may be empty. A caller that wants to *report* what happened needs
    the third one, so this keeps them apart:

    - ``live``    — posts came back.
    - ``empty``   — Reddit answered 200 with no entries, every time we asked.
                    Sometimes genuinely no results, sometimes a soft block; the
                    feed looks identical either way, hence ``suspect_empty``.
    - ``failed``  — could not be fetched at all; ``error`` says why.
    """

    subreddit: str
    keyword: str
    posts: list[RedditPost] = field(default_factory=list)
    error: str = ""
    empty_attempts: int = 0

    @property
    def status(self) -> str:
        if self.error:
            return "failed"
        return "live" if self.posts else "empty"

    @property
    def ok(self) -> bool:
        return self.status == "live"

    @property
    def suspect_empty(self) -> bool:
        """An empty feed we asked for more than once and never got."""
        return self.status == "empty" and self.empty_attempts > 1

    def summary(self) -> str:
        """One line saying what happened, for a log or a status area."""
        where = f"r/{self.subreddit} / {self.keyword!r}"
        if self.status == "failed":
            return f"{where}: failed — {self.error}"
        if self.status == "empty":
            asked = f" after {self.empty_attempts} tries" if self.empty_attempts > 1 else ""
            return f"{where}: answered, but the feed was empty{asked}"
        return f"{where}: {len(self.posts)} posts"


class RedditClient:
    """Throttled reader for Reddit's Atom feeds. Reuse one instance."""

    def __init__(
        self,
        min_interval: float = 1.0,
        max_retries: int = 4,
        timeout: float = 20.0,
        backoff: float = 1.5,
        empty_retries: int = 2,
    ) -> None:
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff = backoff
        self.empty_retries = empty_retries
        self._lock = threading.Lock()
        self._last_request = 0.0

    def get(self, url: str) -> bytes:
        """Fetch a URL, throttled, retrying on 429/5xx."""
        headers = {
            "User-Agent": BROWSER_UA,
            "Accept": "application/atom+xml, text/xml;q=0.9, */*;q=0.8",
            "Accept-Encoding": "gzip",
        }
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            with self._lock:  # throttle
                pause = self.min_interval - (time.monotonic() - self._last_request)
                if pause > 0:
                    time.sleep(pause)
                self._last_request = time.monotonic()

            # A fresh token per attempt, so a retry is never a repeated URL.
            token = secrets.token_hex(8)
            joiner = "&" if "?" in url else "?"
            request = urllib.request.Request(
                f"{url}{joiner}feed={token}&user=u{token[:8]}", headers=headers
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read()
                    if response.headers.get("Content-Encoding") == "gzip":
                        body = gzip.decompress(body)
                    return body
            except urllib.error.HTTPError as exc:
                last_error = exc
                if not (exc.code == 429 or 500 <= exc.code < 600):
                    break
                if attempt < self.max_retries - 1:
                    time.sleep(self._pause_for(exc, attempt))
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff * (2**attempt))

        raise RedditError(f"Could not fetch {url}: {last_error}") from last_error

    def _pause_for(self, exc: urllib.error.HTTPError, attempt: int) -> float:
        """Prefer Reddit's own Retry-After, else back off exponentially."""
        retry_after = exc.headers.get("Retry-After") if exc.headers else None
        if retry_after:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass
        return self.backoff * (2**attempt)

    def search(
        self,
        subreddit: str,
        keyword: str,
        limit: int = 25,
        sort: str = "relevance",
        time_filter: str = "all",
    ) -> list[RedditPost]:
        """Search one subreddit and return the first page of results."""
        subreddit, url = self._search_url(subreddit, keyword, limit, sort, time_filter)
        return self._search(url, subreddit, limit)[0]

    @staticmethod
    def _search_url(
        subreddit: str, keyword: str, limit: int, sort: str, time_filter: str
    ) -> tuple[str, str]:
        """Validate the arguments and build the feed URL. -> (subreddit, url)"""
        if sort not in SORTS:
            raise ValueError(f"sort must be one of {SORTS}")
        if time_filter not in WINDOWS:
            raise ValueError(f"time_filter must be one of {WINDOWS}")
        subreddit = subreddit.strip().lstrip("/").removeprefix("r/").strip("/")
        if not subreddit:
            raise ValueError("subreddit must not be empty")

        query = urllib.parse.urlencode(
            {"q": keyword, "restrict_sr": "1", "sort": sort, "t": time_filter,
             "limit": max(1, min(int(limit), 100))}
        )
        return subreddit, f"{SEARCH_URL.format(subreddit=urllib.parse.quote(subreddit))}?{query}"

    def _search(self, url: str, subreddit: str, limit: int) -> tuple[list[RedditPost], int]:
        """Fetch and parse, returning the posts and how many times we asked.

        The attempt count is returned rather than stored on the client: two
        Streamlit threads can share one client, and an attribute would race.
        """
        # An empty feed is usually a soft failure, so retry before believing it.
        posts: list[RedditPost] = []
        attempts = 0
        for attempt in range(self.empty_retries + 1):
            attempts = attempt + 1
            posts = parse_search_feed(self.get(url), subreddit)
            if posts:
                break
            if attempt < self.empty_retries:
                time.sleep(self.backoff)
        return posts[:limit], attempts

    def search_result(
        self,
        subreddit: str,
        keyword: str,
        limit: int = 25,
        sort: str = "relevance",
        time_filter: str = "all",
    ) -> SearchResult:
        """``search()`` without the raise — the failure lands in the result.

        Same request, same throttle, same retries; only the reporting differs.
        Use this when you want to show the user what happened per search
        instead of stopping at the first one that breaks.
        """
        try:
            name, url = self._search_url(subreddit, keyword, limit, sort, time_filter)
            posts, attempts = self._search(url, name, limit)
        except (RedditError, ValueError) as exc:
            return SearchResult(subreddit=subreddit, keyword=keyword, error=str(exc))
        return SearchResult(
            subreddit=name,
            keyword=keyword,
            posts=posts,
            empty_attempts=0 if posts else attempts,
        )


def parse_search_feed(xml_bytes: bytes, subreddit: str = "") -> list[RedditPost]:
    """Turn a search.rss Atom feed into RedditPost objects."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        preview = xml_bytes[:120].decode("utf-8", "replace")
        raise RedditError(
            f"Reddit did not return a feed (got {preview!r}...) — usually blocked "
            "or rate-limited."
        ) from exc

    posts = []
    for entry in root.findall("atom:entry", ATOM):
        link = entry.find("atom:link", ATOM)
        url = link.get("href", "") if link is not None else ""
        raw_id = (entry.findtext("atom:id", "", ATOM) or "").strip()
        posts.append(
            RedditPost(
                id=raw_id.removeprefix("t3_") or _match(r"/comments/([a-z0-9]+)", url),
                subreddit=subreddit or _match(r"/r/([^/]+)/", url),
                title=(entry.findtext("atom:title", "", ATOM) or "").strip(),
                author=(entry.findtext("atom:author/atom:name", "", ATOM) or "").strip(),
                url=url,
                created_utc=_parse_time(
                    entry.findtext("atom:published", "", ATOM)
                    or entry.findtext("atom:updated", "", ATOM)
                ),
            )
        )
    return posts


def search_subreddit(
    subreddit: str,
    keyword: str,
    limit: int = 25,
    sort: str = "relevance",
    time_filter: str = "all",
    client: RedditClient | None = None,
) -> list[RedditPost]:
    """Search one subreddit. Raises RedditError if it cannot be fetched.

    Pass a shared ``client`` across calls so one throttle covers them all.
    """
    return (client or RedditClient()).search(subreddit, keyword, limit, sort, time_filter)


def search_subreddits(
    keyword: str,
    subreddits: Sequence[str],
    limit: int = 25,
    sort: str = "relevance",
    time_filter: str = "all",
    client: RedditClient | None = None,
) -> tuple[dict[str, list[RedditPost]], dict[str, str]]:
    """Search several subreddits. Returns (results, errors).

    A subreddit that fails lands in errors rather than raising, so one dead
    subreddit never sinks the comparison.
    """
    if isinstance(subreddits, str):
        raise TypeError("subreddits must be a list of names, not a single string")

    client = client or RedditClient()
    results: dict[str, list[RedditPost]] = {}
    errors: dict[str, str] = {}
    for subreddit in subreddits:
        try:
            results[subreddit] = client.search(subreddit, keyword, limit, sort, time_filter)
        except (RedditError, ValueError) as exc:
            errors[subreddit] = str(exc)
    return results, errors


def health_check(
    client: RedditClient | None = None,
    limit: int = 5,
    sort: str = "new",
    time_filter: str = "all",
) -> SearchResult:
    """Probe live access with a throwaway query, unrelated to the user's search.

    Pass the ``limit``/``sort``/``time_filter`` you are about to search with,
    and the same ``client``. The probe is only predictive if it costs Reddit
    what the real searches cost: a 5-post ``new`` query answers when a 25-post
    ``relevance`` query is still being refused, so a cheap probe reports a
    health the searches behind it do not have.
    """
    probe = (client or RedditClient()).search_result(
        "AskReddit", "today", limit=limit, sort=sort, time_filter=time_filter
    )
    return probe


def _parse_time(value: str | None) -> datetime | None:
    """Parse Reddit's '2026-01-14T06:56:51+00:00' timestamps."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _match(pattern: str, text: str) -> str:
    found = re.search(pattern, text or "", re.IGNORECASE)
    return found.group(1) if found else ""


if __name__ == "__main__":
    # Try it without writing any code:
    #     python scraper.py                     -> the default demo query
    #     python scraper.py messi Barca soccer  -> your own
    import argparse

    parser = argparse.ArgumentParser(description="Search subreddits for a keyword.")
    parser.add_argument("keyword", nargs="?", default="neymar")
    parser.add_argument("subreddits", nargs="*", default=["Barca", "psg"])
    parser.add_argument("--limit", type=int, default=5, help="posts per subreddit")
    parser.add_argument("--sort", default="relevance", choices=SORTS)
    parser.add_argument("--time", dest="time_filter", default="all", choices=WINDOWS)
    args = parser.parse_args()

    # One client, so the probe and the searches share a throttle -- and the
    # probe is sent with the same weight as the searches, so it predicts them.
    client = RedditClient()
    probe = health_check(
        client=client, limit=args.limit, sort=args.sort, time_filter=args.time_filter
    )
    print(f"probe: {probe.summary()}")
    found, failed = search_subreddits(
        args.keyword, args.subreddits, args.limit, args.sort, args.time_filter,
        client=client,
    )
    for name, posts in found.items():
        print(f"\nr/{name}: {len(posts)} posts about {args.keyword!r}")
        for post in posts:
            print(f"  [{post.recency:>14}]  {post.title[:70]}")
    for name, error in failed.items():
        print(f"\nr/{name} FAILED: {error}")
