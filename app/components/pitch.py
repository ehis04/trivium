"""Team 6-by-3 zone share maps on the StatsBomb pitch scale (team level, no players)."""

from __future__ import annotations

import io
from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from trivium.charts.pitch import (
    SHARE_CMAP,
    SHARE_NORM,
    _usable,
    zone_figure,
)


def zone_map(table: pd.DataFrame, measure: str = "territory_share") -> None:
    """One team's zones (one match or summed over the tournament); the team attacks left to right."""
    zone_panels([("", table)], measure)


def zone_panels(panels: Sequence[tuple[str, pd.DataFrame]], measure: str = "territory_share") -> None:
    """Titled zone maps side by side, e.g. the selected match and the team's tournament."""
    usable = tuple((title, table) for title, table in panels if _usable(table, measure))
    if not usable:
        st.info("No zone shares are available for this selection.")
        return
    st.image(_zone_png(usable, measure), width="stretch")
    measure_name = "located passes and carries" if measure == "territory_share" else "contests"
    st.caption(
        f"Each of the 18 boxes is a pitch zone, read from the team's own goal toward its attack; the percentage "
        f"shows its share of {measure_name}, and n is the number of actions. Blue means the team had more, orange "
        "means the opponent had more, and grey means they were even; more territory in attacking zones or more "
        "contests won can help a team control play."
    )


@st.cache_data(show_spinner=False, max_entries=64)
def _zone_png(panels: tuple[tuple[str, pd.DataFrame], ...], measure: str) -> bytes:
    """The zone maps side by side as PNG bytes (cached, so a rerun does not redraw them)."""
    figure = zone_figure(panels, measure)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110)
    plt.close(figure)
    return buffer.getvalue()
