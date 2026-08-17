"""Real captured responses, used only when a live search fails.

Not mock data: the genuine output of search_subreddits at capture time, stored
per (keyword, subreddit) pair. Refresh with ``python snapshot.py``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from scraper import RedditClient, RedditError, search_subreddits

SNAPSHOT_PATH = Path(__file__).parent / "data" / "fallback.json"

DEFAULT_KEYWORDS = ["neymar", "messi"]
DEFAULT_SUBREDDITS = ["Barca", "psg", "Brazil"]


def save(
    keywords: list[str] | None = None,
    subreddits: list[str] | None = None,
    limit: int = 25,
    path: Path = SNAPSHOT_PATH,
) -> dict:
    """Run every keyword x subreddit search live and write the results."""
    keywords = keywords or list(DEFAULT_KEYWORDS)
    subreddits = subreddits or list(DEFAULT_SUBREDDITS)
    client = RedditClient()  # one client, so the throttle applies across the grid

    captured: dict[str, dict[str, list[dict]]] = {}
    for keyword in keywords:
        results, errors = search_subreddits(
            keyword, subreddits, limit=limit, client=client
        )
        if errors:
            raise RedditError(f"Refusing to save a partial snapshot; {keyword!r} failed: {errors}")
        captured[keyword] = {sub: [p.to_dict() for p in posts] for sub, posts in results.items()}

    if not any(posts for by_sub in captured.values() for posts in by_sub.values()):
        raise RedditError("Refusing to save an empty snapshot.")

    data = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "keywords": keywords,
        "subreddits": subreddits,
        "results": captured,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def load(path: Path = SNAPSHOT_PATH) -> dict | None:
    """Read the saved snapshot, or None if there is not a usable one."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def captured_label(snapshot: dict | None) -> str:
    """When the snapshot was taken, e.g. '2026-08-13 21:31 UTC (2 days old)'."""
    captured = (snapshot or {}).get("captured_at", "")
    try:
        when = datetime.fromisoformat(captured)
    except (TypeError, ValueError):
        return captured or "an unknown time"
    days = (datetime.now(timezone.utc) - when).days
    return f"{when:%Y-%m-%d %H:%M UTC} ({days} days old)"


def posts_for(snapshot: dict | None, keyword: str, subreddit: str) -> list[dict] | None:
    """Posts for this exact keyword and subreddit, else None.

    The keyword lock matters: a snapshot of "neymar" must never stand in for a
    search for "messi", however badly the live request failed.

    >>> snap = {"results": {"neymar": {"Barca": [{"id": "x"}]}}}
    >>> len(posts_for(snap, "neymar", "barca"))
    1
    >>> posts_for(snap, "messi", "Barca") is None
    True
    """
    if not snapshot:
        return None
    wanted_keyword = (keyword or "").strip().lower()
    wanted_subreddit = (subreddit or "").strip().lower()

    for saved_keyword, by_subreddit in (snapshot.get("results") or {}).items():
        if saved_keyword.strip().lower() != wanted_keyword:
            continue
        for saved_subreddit, posts in (by_subreddit or {}).items():
            if saved_subreddit.strip().lower() == wanted_subreddit:
                return posts
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keywords", nargs="+", default=DEFAULT_KEYWORDS)
    parser.add_argument("--subreddits", nargs="+", default=DEFAULT_SUBREDDITS)
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()

    data = save(keywords=args.keywords, subreddits=args.subreddits, limit=args.limit)
    total = sum(len(p) for by_sub in data["results"].values() for p in by_sub.values())
    print(f"Saved {total} posts to {SNAPSHOT_PATH}")
    for keyword, by_sub in data["results"].items():
        counts = ", ".join(f"r/{sub}: {len(posts)}" for sub, posts in by_sub.items())
        print(f"  {keyword}: {counts}")
