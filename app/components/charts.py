"""Small Plotly figures used by pages where no package chart already exists."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import trivium.charts  # noqa: F401  (visible reference lines in the simple_white template)

ACCENT = "#0072B2"
GREY = "#777777"
LIGHT_GREY = "#B8B8B8"


def comparison_chart(table: pd.DataFrame, label_column: str, value_column: str, average: float) -> go.Figure:
    """Compare one team's values with tournament averages."""
    fig = px.bar(
        table,
        x=value_column,
        y=label_column,
        orientation="h",
        color_discrete_sequence=[ACCENT],
        labels={value_column: "Average per match", label_column: ""},
        template="simple_white",
    )
    fig.add_vline(x=average, line_color=GREY, line_dash="dash")
    fig.update_layout(font={"size": 14}, showlegend=False, margin={"l": 8, "r": 15, "t": 8, "b": 45})
    return fig


def match_trend(table: pd.DataFrame) -> go.Figure:
    """Show match-by-match over-performance and dominance for one team."""
    fig = go.Figure()
    for column, label, color in (
        ("overperformance", "Points above expectation", ACCENT),
        ("dominance", "Overall dominance", GREY),
    ):
        if column in table and table[column].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=table["match_date"] if "match_date" in table else table["match_id"],
                    y=table[column],
                    mode="lines+markers",
                    name=label,
                    line={"color": color},
                    customdata=table[["opponent", "competition_stage"]].to_numpy()
                    if {"opponent", "competition_stage"}.issubset(table.columns)
                    else None,
                    hovertemplate=(
                        "%{x}<br>%{y:.2f}<br>Opponent: %{customdata[0]}"
                        "<br>Stage: %{customdata[1]}<extra></extra>"
                        if {"opponent", "competition_stage"}.issubset(table.columns)
                        else "%{x}<br>%{y:.2f}<extra></extra>"
                    ),
                )
            )
    fig.update_layout(
        template="simple_white",
        font={"size": 14},
        xaxis_title="Match date",
        yaxis_title="Value",
        margin={"l": 50, "r": 20, "t": 10, "b": 45},
    )
    return fig


def efficiency_rank(table: pd.DataFrame, summary: pd.DataFrame | None = None) -> go.Figure:
    """Show the ranked efficiency index with tooltip breakdown and expectation colouring."""
    if table.empty or "efficiency_index" not in table:
        return go.Figure()
    rows = table.sort_values("efficiency_index", ascending=True).reset_index(drop=True)

    overperf_map = {}
    if summary is not None and "team" in summary and "overperformance" in summary:
        overperf_map = dict(zip(summary["team"], summary["overperformance"]))
    elif "overperformance" in rows:
        overperf_map = dict(zip(rows["team"], rows["overperformance"]))

    colors = [
        ACCENT if overperf_map.get(team, 0.0) > 0 else LIGHT_GREY
        for team in rows["team"]
    ]

    customdata = []
    for _, row in rows.iterrows():
        cm = str(row.get("cost_measure", ""))
        cv = row.get("cost_value", 0.0)
        cz = row.get("cost_z", 0.0)
        conv = row.get("conversion_value", 0.0)
        conv_z = row.get("conversion_z", 0.0)
        customdata.append([cm, cv, cz, conv, conv_z])

    fig = go.Figure(
        go.Bar(
            x=rows["efficiency_index"],
            y=rows["team"],
            orientation="h",
            marker_color=colors,
            customdata=customdata,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Efficiency index: %{x:.2f}<br>"
                "Cost measure: %{customdata[0]}<br>"
                "Cost value: %{customdata[1]:.2f}<br>"
                "Cost z: %{customdata[2]:.2f}<br>"
                "Conversion value: %{customdata[3]:.2f}<br>"
                "Conversion z: %{customdata[4]:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        template="simple_white",
        font={"size": 14},
        xaxis_title="Efficiency index",
        yaxis_title="Team",
        yaxis={"categoryorder": "array", "categoryarray": rows["team"].tolist()},
        margin={"l": 80, "r": 20, "t": 10, "b": 45},
    )
    return fig


def profile_comparison(table: pd.DataFrame) -> go.Figure:
    """Compare raw progression and expectation-adjusted profile differences."""
    required = {"label", "raw_difference_z", "adjusted_difference_z"}
    if not required.issubset(table.columns):
        return go.Figure()
    fig = go.Figure()
    raw_errors = _interval_errors(table, "raw_difference_z", "raw_low", "raw_high")
    adjusted_errors = _interval_errors(
        table,
        "adjusted_difference_z",
        "adjusted_low",
        "adjusted_high",
    )
    fig.add_trace(
        go.Bar(
            x=table["label"],
            y=table["raw_difference_z"],
            name="Progressing vs exiting",
            marker_color=GREY,
            error_y=raw_errors,
        )
    )
    fig.add_trace(
        go.Bar(
            x=table["label"],
            y=table["adjusted_difference_z"],
            name="Above vs below expectation",
            marker_color=ACCENT,
            error_y=adjusted_errors,
        )
    )
    if "survives_adjustment" in table:
        fig.add_trace(
            go.Scatter(
                x=table["label"],
                y=table["adjusted_difference_z"],
                mode="markers",
                marker={
                    "color": "#1B1B1B",
                    "symbol": [
                        "diamond" if bool(value) else "circle-open"
                        for value in table["survives_adjustment"].fillna(False)
                    ],
                    "size": 10,
                },
                name="Survives adjustment",
                hovertemplate="%{x}<br>Adjusted difference: %{y:.2f} SD<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_color="#999999")
    fig.update_layout(
        template="simple_white",
        font={"size": 14},
        barmode="group",
        xaxis_title="Signifier",
        yaxis_title="Difference from comparison group (standard deviations)",
        margin={"l": 65, "r": 15, "t": 10, "b": 65},
    )
    return fig


def _interval_errors(
    table: pd.DataFrame,
    value_column: str,
    low_column: str,
    high_column: str,
) -> dict[str, object] | None:
    if not {value_column, low_column, high_column}.issubset(table.columns):
        return None
    value = pd.to_numeric(table[value_column], errors="coerce")
    low = pd.to_numeric(table[low_column], errors="coerce")
    high = pd.to_numeric(table[high_column], errors="coerce")
    return {
        "type": "data",
        "symmetric": False,
        "array": (high - value).clip(lower=0).fillna(0),
        "arrayminus": (value - low).clip(lower=0).fillna(0),
        "visible": True,
    }


def phase_conversion(table: pd.DataFrame) -> go.Figure:
    """Compare xG per possession across the phases present in the feature table."""
    required = {"phase", "xg_per_possession"}
    if not required.issubset(table.columns) or table.empty:
        return go.Figure()
    rows = table.sort_values("xg_per_possession", ascending=True)
    fig = px.bar(
        rows,
        x="xg_per_possession",
        y="phase",
        orientation="h",
        color_discrete_sequence=[ACCENT],
        labels={"xg_per_possession": "Expected goals per possession", "phase": "Phase of play"},
        template="simple_white",
    )
    fig.update_layout(font={"size": 14}, showlegend=False)
    return fig
