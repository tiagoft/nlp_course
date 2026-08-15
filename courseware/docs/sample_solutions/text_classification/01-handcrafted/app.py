"""Streamlit UI. Reads top to bottom: inputs, liveness, search, score, show.

    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import snapshot
from charts import anew_heatmap, featuring_heatmap, opinion_heatmap, to_img_tag
from scraper import RedditClient, RedditPost, health_check, search_subreddits
from sentiment import NEUTRAL, AnewMissing, analyse, describe, load_anew

DEFAULT_KEYWORDS: list[str] = ["neymar", "messi"]
DEFAULT_SUBREDDITS: list[str] = ["Barca", "psg", "Brazil"]

st.set_page_config(page_title="Subreddit sentiment about a word", page_icon="📊", layout="wide")


def split_list(raw: str, strip_prefix: str = "") -> list[str]:
    """Comma-separated input -> a de-duplicated list, in the order typed."""
    items = []
    for part in raw.split(","):
        item = part.strip().lstrip("/")
        if strip_prefix:
            item = item.removeprefix(strip_prefix).strip("/")
        if item:
            items.append(item)
    return list(dict.fromkeys(items))


# --- Cached helpers (Streamlit re-runs this file on every interaction) ---


@st.cache_resource
def get_client() -> RedditClient:
    """One client for the session, so its throttle actually applies."""
    return RedditClient()


@st.cache_resource
def get_anew() -> dict[str, float]:
    return load_anew()


@st.cache_data(ttl=60, show_spinner=False)
def check_reddit() -> tuple[bool, str]:
    """Liveness probe, independent of whatever the user searched for."""
    return health_check(client=RedditClient(max_retries=2, backoff=1.0, timeout=10))


@st.cache_data(ttl=300, show_spinner=False)
def run_search(
    keywords: tuple[str, ...], subreddits: tuple[str, ...], limit: int, sort: str, window: str
):
    """Every keyword across every subreddit, keyed [keyword][subreddit].

    Returns plain dicts, which is what Streamlit can cache.
    """
    posts: dict[str, dict[str, list[dict]]] = {}
    errors: dict[str, dict[str, str]] = {}
    for keyword in keywords:
        found, failed = search_subreddits(
            keyword, list(subreddits), limit=limit, sort=sort, time_filter=window,
            client=get_client(),
        )
        posts[keyword] = {sub: [p.to_dict() for p in items] for sub, items in found.items()}
        if failed:
            errors[keyword] = failed
    return posts, errors


# --- Inputs ---

st.sidebar.header("Query")
keywords: list[str] = split_list(
    st.sidebar.text_input(
        "Keywords (comma separated)",
        value=", ".join(DEFAULT_KEYWORDS),
        help="Each keyword becomes a column in the heatmaps.",
    )
)
subreddits: list[str] = split_list(
    st.sidebar.text_input(
        "Subreddits to compare (comma separated)",
        value=", ".join(DEFAULT_SUBREDDITS),
        help="Names only, without the r/ prefix. Each becomes a row.",
    ),
    strip_prefix="r/",
)

st.sidebar.header("Options")
limit: int = st.sidebar.slider("Posts per subreddit", 5, 100, 25, step=5)
sort: str = st.sidebar.selectbox("Sort", ["relevance", "top", "new", "hot", "comments"])
window: str = st.sidebar.selectbox("Time window", ["all", "year", "month", "week", "day", "hour"])
refresh: bool = st.sidebar.button(
    "Fetch fresh data", type="primary", use_container_width=True,
    help="Results are cached for 5 minutes. This drops the cache and re-fetches now.",
)

st.title("What do subreddits think about a word?")
st.caption(
    "Searches each subreddit for each keyword, then scores the post titles with the "
    "ANEW pleasure norms. Higher is more pleasant; 50 is neutral."
)

# --- 1. Is Reddit actually reachable right now? ---

if refresh:
    run_search.clear()
    check_reddit.clear()

live_ok, live_message = check_reddit()
if live_ok:
    st.success(live_message, icon="✅")
else:
    st.error(
        f"{live_message}\n\nAnything below is cached or a saved snapshot, **not** live data.",
        icon="🚨",
    )

if not keywords:
    st.warning("Enter at least one keyword in the sidebar.")
    st.stop()
if not subreddits:
    st.warning("Enter at least one subreddit in the sidebar.")
    st.stop()

try:
    anew = get_anew()
except AnewMissing as exc:
    st.error(f"Could not load the ANEW dictionary: {exc}")
    st.stop()

# --- 2. Search, falling back to the saved snapshot only where a search failed ---

with st.spinner(f"Searching {len(subreddits)} subreddit(s) for {len(keywords)} keyword(s)..."):
    raw_posts, errors = run_search(tuple(keywords), tuple(subreddits), limit, sort, window)

saved = snapshot.load()
posts_by_cell: dict[tuple[str, str], list[RedditPost]] = {}
from_snapshot: list[tuple[str, str]] = []
missing: list[str] = []

for keyword in keywords:
    for sub in subreddits:
        live = (raw_posts.get(keyword) or {}).get(sub)
        # Only consulted when this cell's live search failed, and it returns
        # None unless the snapshot holds this exact keyword.
        fallback = None if live is not None else snapshot.posts_for(saved, keyword, sub)
        source = live if live is not None else fallback
        if not source:
            missing.append(f"r/{sub}/{keyword}")
            continue
        posts_by_cell[(sub, keyword)] = [RedditPost.from_dict(p) for p in source]
        if fallback:
            from_snapshot.append((sub, keyword))

if errors:
    st.error(
        "Live search failed for: "
        + "; ".join(
            f"**r/{sub}** / *{keyword}* ({why})"
            for keyword, failed in errors.items()
            for sub, why in failed.items()
        ),
        icon="🚨",
    )
if from_snapshot:
    cells = ", ".join(f"r/{sub}/{keyword}" for sub, keyword in from_snapshot)
    st.warning(
        f"Showing a **saved snapshot** for {cells} — real Reddit responses captured on "
        f"{snapshot.captured_label(saved)}, not live data. "
        "Run `python snapshot.py` to refresh.",
        icon="⚠️",
    )

if missing:
    st.error(f"No data at all for: {', '.join(missing)}.", icon="🚨")
if not posts_by_cell:
    st.stop()

# --- 3. Score every (subreddit, keyword) cell ---

results = {
    cell: analyse(cell[0], posts, cell[1], anew) for cell, posts in posts_by_cell.items()
}

# --- 4. Show ---

st.subheader("Results")


def show(build, column=None, max_width: int = 460) -> None:
    """Draw one heatmap, optionally inside a column."""
    tag = to_img_tag(build(results, keywords, subreddits), max_width=max_width)
    (column or st).markdown(tag, unsafe_allow_html=True)


show(opinion_heatmap, max_width=620)

# Two across, not three: three would squeeze each to ~250px.
if st.toggle("Show coverage heatmaps", help="How on-topic the posts were, and how much ANEW could read."):
    left, right = st.columns(2)
    show(featuring_heatmap, left)
    show(anew_heatmap, right)

summary = pd.DataFrame(
    [results[(sub, keyword)].summary_row() for keyword in keywords for sub in subreddits
     if (sub, keyword) in results]
)
summary["reading"] = summary["opinion"].map(describe)
for column in ("% featuring", "% with ANEW words"):
    summary[column] = summary[column].map("{:.0%}".format)
st.dataframe(summary, hide_index=True, use_container_width=True)

st.subheader("Posts")
for (sub, keyword), result in results.items():
    label = f"“{keyword}” — r/{sub} — {result.posts_found} posts"
    if (sub, keyword) in from_snapshot:
        label += "  (snapshot)"
    with st.expander(label):
        st.dataframe(
            pd.DataFrame([post.table_row() for post in result.posts]),
            hide_index=True,
            use_container_width=True,
            column_config={"link": st.column_config.LinkColumn("link", display_text="open")},
        )

with st.expander("How the score is computed"):
    ignored = sorted({word for result in results.values() for word in result.ignored_words})
    st.markdown(
        f"""
A title without the keyword is skipped. Otherwise it scores the mean ANEW
pleasure of its words, weighted by how often each occurs, and a subreddit's
opinion is the mean of those scores.

The keyword itself is dropped from the dictionary — it appears in every hit,
so counting it would measure the search term rather than the language around
it. Dropped here: {", ".join(f"`{w}`" for w in ignored) if ignored
else "nothing, since neither keyword is an ANEW word"}.

ANEW rates {len(anew)} words from 0 to 100, so {NEUTRAL:.0f} is neutral.
Posts come from Reddit's Atom feed; the `.json` endpoints answer 403.
"""
    )
