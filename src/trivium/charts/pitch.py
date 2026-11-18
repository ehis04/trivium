"""Pitch zone maps for team territory and contest shares (team level, no player minors).

Draws 6-by-3 zone grids on a StatsBomb pitch with diverging colors centered on 50%
and attacking direction indicator.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mplsoccer import Pitch

ACCENT = "#0072B2"
OPPONENT = "#E69F00"
GREY = "#777777"
LIGHT_GREY = "#B8B8B8"

MEASURES = {
    "territory_share": ("team_actions", "opponent_actions"),
    "contests_share": ("contests_won", "contests_lost"),
}

# Opponent ahead (orange) -> even (light grey, so a 50% zone stays visible) -> team ahead (blue).
SHARE_CMAP = LinearSegmentedColormap.from_list("share", [OPPONENT, "#E3E3E3", ACCENT])
SHARE_NORM = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)


def _usable(table: pd.DataFrame, measure: str) -> bool:
    if measure not in MEASURES:
        raise ValueError("Zone-map measure must be territory_share or contests_share.")
    if table.empty or not {"zone_x", "zone_y", measure}.issubset(table.columns):
        return False
    return pd.to_numeric(table[measure], errors="coerce").notna().any()


def _draw_zone_map(axis: Axes, table: pd.DataFrame, measure: str, font_size: int = 10) -> None:
    """Shaded zone cells with share and count labels on one pitch axis."""
    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="white",
        line_color="#7b8794",
        linewidth=1,
        line_zorder=2,
    )
    pitch.draw(ax=axis)
    won, lost = MEASURES[measure]
    for row in table.itertuples(index=False):
        share = getattr(row, measure)
        if pd.isna(share):
            continue
        n = sum(
            int(getattr(row, c))
            for c in (won, lost)
            if hasattr(row, c) and pd.notna(getattr(row, c))
        )
        x = float(row.zone_x) * 120 / 6
        y = float(row.zone_y) * 80 / 3
        axis.add_patch(
            Rectangle(
                (x, y),
                20,
                80 / 3,
                facecolor=SHARE_CMAP(SHARE_NORM(float(share))),
                edgecolor="white",
                zorder=1,
            )
        )
        axis.text(
            x + 10,
            y + 40 / 3,
            f"{float(share):.0%}\nn={n}",
            ha="center",
            va="center",
            fontsize=font_size,
            zorder=3,
        )
    axis.annotate(
        "",
        xy=(80, 84),
        xytext=(40, 84),
        annotation_clip=False,
        arrowprops={"arrowstyle": "->", "color": "#555555"},
    )
    axis.text(
        60,
        86,
        "Attacking direction",
        ha="center",
        va="top",
        fontsize=font_size,
        color="#555555",
    )


def zone_figure(
    panels: Sequence[tuple[str, pd.DataFrame]],
    measure: str = "territory_share",
    font_size: int = 10,
) -> Figure:
    """Titled zone maps side by side; only panels with available shares are drawn."""
    if measure not in MEASURES:
        raise ValueError("Zone-map measure must be territory_share or contests_share.")
    usable = [(title, table) for title, table in panels if _usable(table, measure)]
    if not usable:
        fig = plt.figure(figsize=(7, 5))
        return fig
    figure, axes = plt.subplots(
        1,
        len(usable),
        figsize=(7 * len(usable), 5),
        squeeze=False,
    )
    title_size = max(13, font_size + 2)
    for (title, table), axis in zip(usable, axes[0]):
        _draw_zone_map(axis, table, measure, font_size=font_size)
        if title:
            axis.set_title(title, fontsize=title_size)
    figure.tight_layout()
    return figure
