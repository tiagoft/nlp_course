"""Streamlit UI. Reads top to bottom: inputs, liveness, search, score, show.

    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import snapshot
from charts import anew_heatmap, featuring_heatmap, opinion_heatmap, to_img_tag
from scraper import RedditClient, RedditPost, SearchResult, health_check
from sentiment import NEUTRAL, AnewMissing, analyse, describe, load_anew

DEFAULT_KEYWORDS: list[str] = ["neymar", "messi"]
DEFAULT_SUBREDDITS: list[str] = ["Barca", "psg", "Brazil"]

# One TTL for the probe and the searches, so the banner at the top can never
# describe a different moment in time than the results underneath it.
SEARCH_TTL = 300
STATUS_ICON = {"live": "✅", "empty": "⚠️", "failed": "🚨"}

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


@st.cache_data(ttl=SEARCH_TTL, show_spinner=False)
def probe_reddit(limit: int, sort: str, window: str) -> SearchResult:
    """Will a search of this size work right now? Sent before we run the real ones.

    It carries the user's own limit/sort/window and goes through the shared
    client, so it costs Reddit exactly what the searches behind it cost. The
    old probe asked for 5 posts sorted by "new" no matter what the user had
    chosen, which is a cheaper request than a 25-post "relevance" search --
    that is how it could report OK while every search below it got a 500.
    """
    return health_check(client=get_client(), limit=limit, sort=sort, time_filter=window)


@st.cache_data(ttl=SEARCH_TTL, show_spinner=False)
def search_cell(keyword: str, subreddit: str, limit: int, sort: str, window: str) -> dict:
    """One subreddit x keyword search, cached on its own so it can be reported
    the moment it lands rather than after the whole grid finishes.

    Returns plain dicts, which is what Streamlit can cache.
    """
    result = get_client().search_result(
        subreddit, keyword, limit=limit, sort=sort, time_filter=window
    )
    return {
        "posts": [post.to_dict() for post in result.posts],
        "status": result.status,
        "error": result.error,
        "suspect_empty": result.suspect_empty,
        "summary": result.summary(),
    }


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

# --- 1. Before searching: will a search this size work right now? ---

if refresh:
    search_cell.clear()
    probe_reddit.clear()

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

cells_total = len(keywords) * len(subreddits)
shape = f"sorted by {sort}, {limit} posts, window {window}"

with st.spinner(f"Checking whether Reddit will answer a search this size ({shape})..."):
    probe = probe_reddit(limit, sort, window)

if probe.ok:
    st.success(
        f"**Pre-flight OK** — a throwaway probe with your exact settings ({shape}) came "
        f"back with {len(probe.posts)} posts, so the {cells_total} searches below should "
        "work. One probe is evidence, not a promise: Reddit refuses request by request, "
        "so individual subreddits can still fail. Each is reported live as it lands.",
        icon="✅",
    )
elif probe.status == "empty":
    st.warning(
        f"**Pre-flight unclear** — Reddit answered the probe ({shape}) but sent an empty "
        "feed, which is what a soft block looks like from here. The searches below may "
        "come back empty for the same reason rather than because nothing matched.",
        icon="⚠️",
    )
else:
    st.error(
        f"**Pre-flight failed** — Reddit refused a probe with your exact settings "
        f"({shape}): {probe.error}\n\nExpect the {cells_total} searches below to fail the "
        "same way and fall back to the saved snapshot. Lowering **Posts per subreddit** "
        "is the setting most likely to get through: a cheap request is served when an "
        "expensive one is refused.",
        icon="🚨",
    )

# --- 2. Search, reporting each cell as it lands ---

outcomes: dict[tuple[str, str], dict] = {}
with st.status(f"Searching {cells_total} subreddit × keyword pair(s)...", expanded=True) as box:
    for keyword in keywords:
        for sub in subreddits:
            cell = search_cell(keyword, sub, limit, sort, window)
            outcomes[(sub, keyword)] = cell
            st.write(f"{STATUS_ICON[cell['status']]} {cell['summary']}")
    live_count = sum(1 for cell in outcomes.values() if cell["status"] == "live")
    # Collapsed once it is done either way: the per-cell lines are the detail
    # behind the banners below, and left open they bury the results in URLs.
    box.update(
        label=f"{live_count}/{cells_total} searches came back live",
        state="complete" if live_count == cells_total else "error",
        expanded=False,
    )

# --- 3. Fall back to the snapshot where the live data did not arrive ---

saved = snapshot.load()
posts_by_cell: dict[tuple[str, str], list[RedditPost]] = {}
from_snapshot: list[tuple[str, str]] = []
missing: list[str] = []

for keyword in keywords:
    for sub in subreddits:
        cell = outcomes[(sub, keyword)]
        live = cell["posts"]
        # A feed that stayed empty across every retry is as likely a soft block
        # as a genuine absence of posts, so it gets the snapshot too. Only
        # consulted for this exact keyword, so a "neymar" capture can never
        # stand in for a "messi" search.
        fallback = (
            snapshot.posts_for(saved, keyword, sub)
            if cell["status"] == "failed" or cell["suspect_empty"]
            else None
        )
        source = live or fallback
        if not source:
            missing.append(f"r/{sub}/{keyword}")
            continue
        posts_by_cell[(sub, keyword)] = [RedditPost.from_dict(p) for p in source]
        if not live and fallback:
            from_snapshot.append((sub, keyword))

failed = [(sub, kw) for (sub, kw), cell in outcomes.items() if cell["status"] == "failed"]
if failed:
    st.error(
        "Live search failed for: "
        + "; ".join(f"**r/{sub}** / *{keyword}*" for sub, keyword in failed)
        + ("  \nThe pre-flight probe said this would happen." if not probe.ok
           else "  \nThe pre-flight probe got through, so this is per-subreddit, not a "
                "general block."),
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

st.caption(
    f"{live_count} live · {len(from_snapshot)} snapshot · {len(missing)} unavailable "
    f"— out of {cells_total} pair(s)."
)

# --- 4. Score every (subreddit, keyword) cell ---

results = {
    cell: analyse(cell[0], posts, cell[1], anew) for cell, posts in posts_by_cell.items()
}

# --- 5. Show ---

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
