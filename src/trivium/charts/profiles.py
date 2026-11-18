"""Plotly figure for the raw vs adjusted profile comparison (supplement addition 4).

Returns: a figure with one row per measure, the raw difference (progressing minus exiting)
and the adjusted difference (above minus below Elo expectation), each with its interval.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

ACCENT = "#0072B2"
GREY = "#777777"
LIGHT_GREY = "#B8B8B8"


def _intervals(rows: pd.DataFrame, kind: str) -> dict[str, object]:
    return {"type": "data", "symmetric": False, "array": rows[f"{kind}_high"] - rows[f"{kind}_difference_z"],
            "arrayminus": rows[f"{kind}_difference_z"] - rows[f"{kind}_low"], "thickness": 1.5}


def profile_comparison_chart(comparison: pd.DataFrame, tournament: str) -> go.Figure:
    """One row per measure, sorted by the adjusted difference: hollow grey = raw (progressing minus
    exiting), filled = adjusted (above minus below Elo expectation); adjusted rows whose interval
    crosses zero are faded."""
    rows = comparison[comparison["tournament"] == tournament].sort_values(
        "adjusted_difference_z", ignore_index=True)
    fig = go.Figure()
    fig.update_layout(template="simple_white", font={"size": 14})
    if rows.empty:
        return fig
    clear = rows["adjusted_excludes_zero"].fillna(False).astype(bool)
    rows["label"] = [f"<b>{label}</b>" if c else label for label, c in zip(rows["label"], clear)]
    fig.add_trace(go.Scatter(
        x=rows["raw_difference_z"], y=rows["label"], mode="markers", name="Progressing minus exiting (raw)",
        marker={"color": GREY, "symbol": "circle-open", "size": 11, "line": {"color": GREY, "width": 2}},
        error_x=_intervals(rows, "raw") | {"color": LIGHT_GREY},
        hovertemplate="%{y}<br>Raw: %{x:.2f} standard deviations<extra></extra>"))
    for subset, colour, name in ((rows[clear], ACCENT, "Above minus below expectation (adjusted): clear of zero"),
                                 (rows[~clear], LIGHT_GREY, "Adjusted: interval crosses zero")):
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset["adjusted_difference_z"], y=subset["label"], mode="markers", name=name,
            marker={"color": colour, "size": 12}, error_x=_intervals(subset, "adjusted") | {"color": colour},
            hovertemplate="%{y}<br>Adjusted: %{x:.2f} standard deviations<extra></extra>"))
    fig.add_vline(x=0, line_color=GREY, line_dash="dash")
    fig.update_layout(
        margin={"l": 260, "r": 20, "t": 60, "b": 60},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0},
        yaxis={"categoryorder": "array", "categoryarray": rows["label"].tolist(), "title": ""},
        xaxis_title="Difference in team averages (standard deviations, 95% interval)",
    )
    return fig
