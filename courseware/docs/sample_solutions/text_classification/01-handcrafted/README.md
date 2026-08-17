# Sentiment analysis with ANEW

Searches subreddits for one or more keywords and scores the post titles with
the ANEW pleasure norms, comparing every subreddit × keyword pair.

```bash
pip install -r requirements.txt
streamlit run app.py
```

| file | |
|---|---|
| `app.py` | Streamlit UI |
| `sentiment.py` | the scoring |
| `charts.py` | the heatmaps |
| `snapshot.py` | saved responses, used when a live request fails |
| `scraper.py` | fetching Reddit |

## Using the scraper in your own project

Downloading the posts is where most of the time goes: Reddit's unauthenticated
`.json` endpoints answer 403 now, and `old.reddit.com` redirects to a login
wall. `scraper.py` deals with that, and is written to be copied out of here —
**one file, Python 3.9+, standard library only, no API key, nothing to
install.**

Copy `scraper.py` next to your own code and import it:

```python
from scraper import search_subreddit, search_subreddits

posts = search_subreddit("Barca", "neymar")   # -> list[RedditPost], raises on failure
posts[0].title, posts[0].recency, posts[0].created_utc, posts[0].url, posts[0].author

results, errors = search_subreddits("neymar", ["Barca", "psg"])
# results: {'Barca': [...], 'psg': [...]}
# errors:  {} — a subreddit that fails lands here instead of raising
```

Both accept `limit`, `sort` (`relevance`, `hot`, `top`, `new`, `comments`) and
`time_filter` (`all`, `hour`, `day`, `week`, `month`, `year`). One search
returns one page: 25 posts, metadata only — no comments, no scores. Doing many
searches? Build one `RedditClient()` and pass it to each call, so a single
throttle covers them all.

It also runs on its own, which is the quickest way to check it still works:

```bash
python scraper.py neymar Barca psg
```

### Why it looks odd

It reads Reddit's RSS feeds rather than JSON, and works around two things I
measured — worth knowing before you tidy them away:

- **Repeated URLs get rate-limited.** The plain search URL returned 200 once in
  six tries, and six times out of six once a throwaway
  `feed=<random>&user=<random>` pair was added to each request.
- **Bot user agents are blocked**, so requests claim to be a browser.

A 200 is also sometimes an empty feed, so empty results are retried before
being believed.

## Scoring

A title without the keyword scores nothing. Otherwise its score is the mean
ANEW pleasure of its words, weighted by how often each occurs, and a
subreddit's opinion is the mean of those scores.

The keyword itself is dropped from the dictionary: it appears in every hit by
construction, so counting it would measure the search term rather than the
language around it. Searching `love` scores 94.6 with the word left in, 86.2
with it dropped.

Pleasure runs 0–100, so 50 is neutral. The dictionary downloads once into
`data/anew.csv`.

Two extra heatmaps, behind a toggle, say how far to trust each cell: how many
posts really contain the keyword, and how many of those had any ANEW word.
r/Brazil returns plenty of hits for *messi*, but only 20% put it in the title.

## Data

`data/fallback.json` holds real captured responses, shown only when a live
request fails, matched to the keyword they were captured for, and always
labelled on screen. Refresh with `python snapshot.py`.

## Saying up front whether it will work

Reddit refuses requests one at a time, so before running the grid the app
sends one throwaway probe and reports what it means:

- **Pre-flight OK** — a probe got through, so the searches probably will too.
- **Pre-flight failed** — expect the grid to fail and fall back to snapshots.

The probe only predicts if it costs Reddit what the real searches cost, so it
carries the **same limit, sort and time window the user chose**, through the
**same client** — the same throttle and the same retry policy. An earlier
version asked for 5 posts sorted by `new` regardless of the settings, and a
cheap request like that is served while a 25-post `relevance` search is still
being refused: the banner read *Live Reddit access OK* above a page of 500s.

A probe is one request, though, not a promise, so each subreddit × keyword
search also reports itself as it lands — `live`, `empty` or `failed` — and the
totals below the fold say how the grid actually went. `SearchResult` in
`scraper.py` is what carries that per-search verdict; `search_result()` is
`search()` without the raise, for callers that would rather report a failure
than stop at it.

One outcome is deliberately not treated as data: a feed that comes back empty
every time we ask. Reddit sends an empty feed both when nothing matched and
when it is soft-blocking, and the two are identical from here, so those cells
fall back to the snapshot too rather than being drawn as a genuine zero.
