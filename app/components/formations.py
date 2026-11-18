"""Cached pitch images for formation shapes."""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from three_trees.charts.formations import average_shape


@st.cache_data(show_spinner=False, max_entries=64)
def shape_png(
    positions_window: pd.DataFrame,
    team: str,
    match_id: int,
    state: str,
    window: tuple[int, int] | None,
) -> bytes:
    """Render a selected team shape as cached PNG bytes."""
    figure = average_shape(positions_window, team, match_id, state, window)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110)
    plt.close(figure)
    return buffer.getvalue()
