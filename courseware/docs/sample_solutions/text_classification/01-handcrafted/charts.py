"""Three heatmaps: keywords as rows, subreddits as columns.

That way round because there are usually more subreddits than keywords, so the
grid stays wide rather than tall. All three share one ``heatmap`` function.
Every cell prints its number, so the colour never has to carry the reading --
worth keeping if you change the ramps, since red/green is the worst pair for
colourblind readers.
"""

from __future__ import annotations

import base64
import io
from typing import Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, FAINT, NO_DATA = "#0b0b0b", "#898781", "#f0efec"
LABEL = "#000000"  # row/column names: pure black, larger and bold
LABEL_SIZE = 13


def to_img_tag(fig, scale: int = 3, max_width: int = 460) -> str:
    """Figure -> an ``<img>`` tag with the PNG inlined as a data URI.

    Drawn at ``scale``x and shown at CSS width, so it stays sharp on a high-DPI
    screen; ``st.image(width=...)`` resamples on the server and loses that.
    ``width:100%`` keeps it inside a narrow column instead of overflowing.
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100 * scale, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        f'<img src="data:image/png;base64,{encoded}" '
        f'style="width:100%;max-width:{max_width}px;height:auto;display:block;'
        f'margin:0 0 1.5rem">'  # breathing room before whatever follows
    )


def heatmap(
    values: np.ndarray,
    rows: Sequence[str],
    cols: Sequence[str],
    title: str,
    cmap: str,
    fmt: str,
    subtitle: str = "",
):
    """One grid. ``values`` is (len(rows), len(cols)); NaN means "no data".

    Limits are fixed at 0-100 so cells stay comparable between runs.
    """
    n_rows, n_cols = len(rows), len(cols)
    fig, ax = plt.subplots(figsize=(1.55 * n_cols + 2.3, 0.55 * n_rows + 1.75))

    palette = matplotlib.colormaps[cmap].copy()
    palette.set_bad(NO_DATA)
    image = ax.imshow(
        np.ma.masked_invalid(values), cmap=palette, vmin=0, vmax=100, aspect="auto"
    )

    ax.set_xticks(
        range(n_cols), labels=cols, fontsize=LABEL_SIZE, fontweight="bold", color=LABEL
    )
    ax.set_yticks(
        range(n_rows), labels=rows, fontsize=LABEL_SIZE, fontweight="bold", color=LABEL
    )
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    # Thin white gutters, so the grid reads as cells rather than one wash.
    ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    for row in range(n_rows):
        for col in range(n_cols):
            value = values[row, col]
            if np.isnan(value):
                ax.text(col, row, "—", ha="center", va="center", fontsize=11, color=FAINT)
                continue
            # Flip the label to white once the cell behind it goes dark.
            red, green, blue, _ = image.cmap(image.norm(value))
            dark = 0.299 * red + 0.587 * green + 0.114 * blue < 0.55
            ax.text(col, row, fmt.format(value), ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white" if dark else INK)

    # The subtitle sits just above the grid, so the title needs pad to clear it.
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=26 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9, color=FAINT,
                va="bottom")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig


def _grid(results: dict, keywords: Sequence[str], subreddits: Sequence[str], measure) -> np.ndarray:
    """Pull one measure out of the {(subreddit, keyword): result} mapping.

    Rows are keywords, columns are subreddits.
    """
    values = np.full((len(keywords), len(subreddits)), np.nan)
    for row, keyword in enumerate(keywords):
        for col, subreddit in enumerate(subreddits):
            result = results.get((subreddit, keyword))
            if result is not None:
                found = measure(result)
                values[row, col] = np.nan if found is None else found
    return values


def opinion_heatmap(results: dict, keywords: Sequence[str], subreddits: Sequence[str]):
    """ANEW pleasure per cell. Red is unpleasant, green pleasant."""
    return heatmap(
        _grid(results, keywords, subreddits, lambda r: r.opinion),
        keywords, [f"r/{sub}" for sub in subreddits],
        "How pleasant is the language around each keyword?",
        cmap="RdYlGn", fmt="{:.1f}",
        subtitle="ANEW pleasure, 0–100 · 50 is neutral · grey means no score",
    )


def featuring_heatmap(results: dict, keywords: Sequence[str], subreddits: Sequence[str]):
    """Share of posts found whose title actually features the keyword."""
    return heatmap(
        _grid(results, keywords, subreddits, lambda r: r.proportion_featuring_keyword * 100),
        keywords, [f"r/{sub}" for sub in subreddits],
        "Of the posts found, how many actually say the keyword?",
        cmap="Greys", fmt="{:.0f}%",
        subtitle="Higher means the search returned more on-topic posts",
    )


def anew_heatmap(results: dict, keywords: Sequence[str], subreddits: Sequence[str]):
    """Share of keyword-featuring posts that had an ANEW word to score."""
    return heatmap(
        _grid(results, keywords, subreddits, lambda r: r.proportion_scored * 100),
        keywords, [f"r/{sub}" for sub in subreddits],
        "Of the posts featuring the keyword, how many had ANEW words?",
        cmap="Blues", fmt="{:.0f}%",
        subtitle="How much of the matching text ANEW could read · keyword excluded",
    )
