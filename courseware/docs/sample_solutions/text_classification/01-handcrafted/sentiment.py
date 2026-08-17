"""Score Reddit post titles with the ANEW pleasure norms (0-100, 50 neutral).

A title without the keyword scores None. Otherwise it scores the mean pleasure
of its ANEW words, weighted by how often each occurs, and a subreddit's opinion
is the mean of those. The keyword itself is excluded: it appears in every hit,
so counting it would measure the search term rather than the talk around it.
"""

from __future__ import annotations

import csv
import re
import unicodedata
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

ANEW_URL = "https://osf.io/download/cq6ng/"  # from https://osf.io/y6g5b/wiki/anew/
ANEW_PATH = Path(__file__).parent / "data" / "anew.csv"
NEUTRAL = 50.0

_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


class AnewMissing(RuntimeError):
    """The ANEW dictionary could not be read or downloaded."""


def load_anew(path: Path = ANEW_PATH) -> dict[str, float]:
    """Load ANEW as {word: pleasure}, downloading it once if needed."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(ANEW_URL, timeout=60) as response:
                path.write_bytes(response.read())
        except Exception as exc:
            raise AnewMissing(f"Could not download ANEW from {ANEW_URL}: {exc}") from exc

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "pleasure" not in reader.fieldnames:
            raise AnewMissing(f"{path} has no 'pleasure' column (found {reader.fieldnames}).")
        words = {}
        for row in reader:
            try:
                words[row["term"].strip().lower()] = float(row["pleasure"])
            except (TypeError, ValueError, AttributeError, KeyError):
                continue
    if not words:
        raise AnewMissing(f"{path} contained no usable rows.")
    return words


def _flatten(text: str) -> str:
    """Lowercase, accents stripped, so 'Barça' matches 'barca'."""
    bare = "".join(
        ch for ch in unicodedata.normalize("NFKD", text or "") if not unicodedata.combining(ch)
    )
    return bare.lower()


def tokenize(text: str) -> list[str]:
    """Split into lowercase words, keeping internal apostrophes.

    >>> tokenize("Neymar's 2015 heroics -- pure joy!")
    ["neymar's", 'heroics', 'pure', 'joy']
    """
    return _WORD.findall(_flatten(text))


def title_features_keyword(title: str, keyword: str) -> bool:
    """Whole-word match, ignoring case and accents. "Neymar's" counts.

    >>> title_features_keyword("NEYMAR's best moments", "neymar")
    True
    >>> title_features_keyword("Barça vs PSG", "barca")
    True
    """
    haystack, needle = _flatten(title), _flatten(keyword).strip()
    if not needle:
        return False
    pattern = r"\s+".join(re.escape(part) for part in needle.split())
    # \b only works next to word characters, so only add it when the keyword
    # starts and ends with one (otherwise "c++" would never match).
    if re.match(r"\w", needle) and re.search(r"\w$", needle):
        pattern = rf"\b{pattern}\b"
    return re.search(pattern, haystack) is not None


def keyword_words(keyword: str, anew: dict[str, float]) -> set[str]:
    """The keyword's own words that are in ANEW, so they can be excluded."""
    return {word for word in tokenize(keyword) if word in anew}


def title_score(
    title: str, anew: dict[str, float], ignore: set[str] = frozenset()
) -> float | None:
    """Mean ANEW pleasure of a title's words, weighted by how often each occurs.

    >>> anew = {"love": 90.0, "hate": 10.0}
    >>> title_score("love love hate", anew)
    63.333333333333336
    >>> title_score("love and hate", anew, ignore={"love"})
    10.0
    """
    counts = Counter(w for w in tokenize(title) if w in anew and w not in ignore)
    if not counts:
        return None
    return sum(n * anew[w] for w, n in counts.items()) / sum(counts.values())


def subreddit_opinion(scores: Iterable[float | None]) -> float | None:
    """Unweighted mean of the non-None scores."""
    usable = [score for score in scores if score is not None]
    return sum(usable) / len(usable) if usable else None


@dataclass
class PostScore:
    title: str
    url: str
    recency: str
    features_keyword: bool
    score: float | None
    matched_words: dict[str, float] = field(default_factory=dict)

    @property
    def skipped_because(self) -> str:
        if not self.features_keyword:
            return "keyword not in title"
        return "no ANEW words in title" if self.score is None else ""

    def table_row(self) -> dict:
        return {
            "title": self.title,
            "recency": self.recency,
            "features keyword": self.features_keyword,
            "score": self.score,
            "skipped because": self.skipped_because,
            "ANEW words": ", ".join(sorted(self.matched_words)),
            "link": self.url,
        }


@dataclass
class SubredditResult:
    """One cell of the grid: what one subreddit says about one keyword."""

    subreddit: str
    keyword: str
    posts: list[PostScore]
    ignored_words: set[str] = field(default_factory=set)

    @property
    def opinion(self) -> float | None:
        return subreddit_opinion(post.score for post in self.posts)

    @property
    def posts_found(self) -> int:
        return len(self.posts)

    @property
    def posts_featuring_keyword(self) -> int:
        return sum(1 for post in self.posts if post.features_keyword)

    @property
    def posts_scored(self) -> int:
        """Keyword present *and* an ANEW word to score."""
        return sum(1 for post in self.posts if post.score is not None)

    @property
    def proportion_featuring_keyword(self) -> float:
        """Of the posts found, how many actually say the keyword."""
        return self.posts_featuring_keyword / self.posts_found if self.posts else 0.0

    @property
    def proportion_scored(self) -> float:
        """Of the posts featuring the keyword, how many had an ANEW word."""
        if not self.posts_featuring_keyword:
            return 0.0
        return self.posts_scored / self.posts_featuring_keyword

    def summary_row(self) -> dict:
        return {
            "keyword": self.keyword,
            "subreddit": self.subreddit,
            "opinion": self.opinion,
            "posts found": self.posts_found,
            "featuring keyword": self.posts_featuring_keyword,
            "% featuring": self.proportion_featuring_keyword,
            "scored": self.posts_scored,
            "% with ANEW words": self.proportion_scored,
        }


def analyse(subreddit: str, posts: Sequence, keyword: str, anew: dict[str, float]) -> SubredditResult:
    """Score one subreddit's posts for one keyword."""
    ignore = keyword_words(keyword, anew)
    scored = []
    for post in posts:
        title = getattr(post, "title", "") or ""
        features = title_features_keyword(title, keyword)
        scored.append(
            PostScore(
                title=title,
                url=getattr(post, "url", "") or "",
                recency=getattr(post, "recency", "unknown"),
                features_keyword=features,
                score=title_score(title, anew, ignore) if features else None,
                matched_words=(
                    {w: anew[w] for w in tokenize(title) if w in anew and w not in ignore}
                    if features else {}
                ),
            )
        )
    return SubredditResult(subreddit, keyword, scored, ignore)


def describe(score: float | None) -> str:
    """A plain-English reading of a pleasure score."""
    if score is None:
        return "no score"
    for floor, label in ((65, "clearly positive"), (55, "mildly positive"),
                         (45, "roughly neutral"), (35, "mildly negative")):
        if score >= floor:
            return label
    return "clearly negative"
