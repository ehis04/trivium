"""Plotly figures for team over-performance, progression and model calibration.

Returns: Plotly figures with team-level or calibration-bin markers and no embedded titles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ACCENT = "#0072B2"
GREY = "#777777"
LIGHT_GREY = "#B8B8B8"
ABOVE = "#009E73"
BELOW = "#D55E00"
STAGES_WITH_ROUND_OF_32 = [
    "Group",
    "Round of 32",
    "Round of 16",
    "Quarter-final",
    "Semi-final",
    "Final",
    "Champion",
]
STAGES_WITHOUT_ROUND_OF_32 = [
    "Group",
    "Round of 16",
    "Quarter-final",
    "Semi-final",
    "Final",
    "Champion",
]


def _figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="simple_white", font={"size": 14})
    return fig


def _tournament_rows(table: pd.DataFrame, tournament: str) -> pd.DataFrame:
    if "tournament" not in table:
        raise ValueError("Missing column: tournament.")
    return table.loc[table["tournament"].eq(tournament)].copy()


def _require_columns(table: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in table]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}.")


def overperformance_chart(
    team_table: pd.DataFrame,
    tournament: str,
    bands: pd.DataFrame | None = None,
    top_n: int | None = None,
) -> go.Figure:
    """Plot team over-performance with optional simulated luck intervals."""
    _require_columns(team_table, ["tournament", "team", "overperformance"])
    if top_n is not None and (
        not isinstance(top_n, (int, np.integer)) or isinstance(top_n, bool) or top_n < 0
    ):
        raise ValueError("top_n must be a non-negative integer.")
    rows = _tournament_rows(team_table, tournament).sort_values(
        "overperformance", ascending=False, kind="stable", ignore_index=True
    )
    if top_n is not None:
        rows = rows.head(top_n).copy()
    if rows.empty:
        return _figure()

    marker_symbols = ["circle"] * len(rows)
    marker_colors = [GREY] * len(rows)
    rows["_band_low"] = np.nan
    rows["_band_high"] = np.nan
    if bands is not None:
        _require_columns(bands, ["tournament", "team", "band_low", "band_high", "beyond_luck"])
        tournament_bands = _tournament_rows(bands, tournament).set_index("team")
        for index, row in rows.iterrows():
            if row["team"] not in tournament_bands.index:
                continue
            band = tournament_bands.loc[row["team"]]
            if isinstance(band, pd.DataFrame):
                band = band.iloc[0]
            status = band["beyond_luck"]
            if status in ("above", "below"):
                marker_symbols[index] = "diamond"
                marker_colors[index] = ACCENT
            low, high = band["band_low"], band["band_high"]
            if pd.notna(low) and pd.notna(high) and low <= high:
                rows.at[index, "_band_low"] = float(low)
                rows.at[index, "_band_high"] = float(high)

    fig = _figure()
    for _, row in rows.iterrows():
        if pd.notna(row["_band_low"]):
            fig.add_shape(
                type="line",
                x0=row["_band_low"],
                x1=row["_band_high"],
                y0=row["team"],
                y1=row["team"],
                xref="x",
                yref="y",
                line={"color": LIGHT_GREY, "width": 2},
                layer="below",
            )
    fig.add_trace(
        go.Scatter(
            x=rows["overperformance"],
            y=rows["team"],
            mode="markers",
            marker={"color": marker_colors, "symbol": marker_symbols, "size": 11},
            customdata=rows["team"],
            hovertemplate="%{customdata}<br>Over-performance: %{x:.2f} points<extra></extra>",
            name="Teams",
        )
    )
    fig.add_vline(x=0, line_color=GREY, line_dash="dash")
    fig.update_layout(
        showlegend=False,
        margin={"l": 120, "r": 20, "t": 15, "b": 55},
        yaxis={"categoryorder": "array", "categoryarray": rows["team"].tolist()},
        xaxis_title="Points above expectation (tournament total)",
        yaxis_title="Team",
    )
    return fig


def overperformance_dumbbell(
    team_summary: pd.DataFrame,
    tournament: str,
    highlight: str | None = None,
    top: int = 5,
    bottom: int = 5,
) -> go.Figure:
    """Expected (grey) and actual (coloured) tournament points per team, joined by a line coloured by
    direction: the top and bottom teams by over-performance plus the highlighted team, sorted by gap."""
    _require_columns(team_summary, ["tournament", "team", "expected_points", "actual_points", "overperformance"])
    rows = _tournament_rows(team_summary, tournament).sort_values(
        "overperformance", ascending=False, kind="stable", ignore_index=True)
    keep = set(rows.head(top)["team"]) | set(rows.tail(bottom)["team"])
    if highlight is not None:
        keep.add(highlight)
    rows = rows[rows["team"].isin(keep)].reset_index(drop=True)
    fig = _figure()
    if rows.empty:
        return fig
    labels = [f"<b>{team}</b>" if team == highlight else team for team in rows["team"]]
    for label, row in zip(labels, rows.itertuples(index=False)):
        colour = ABOVE if row.overperformance >= 0 else BELOW
        fig.add_trace(go.Scatter(
            x=[row.expected_points, row.actual_points], y=[label, label], mode="lines",
            line={"color": colour, "width": 5 if row.team == highlight else 3}, opacity=1.0 if row.team == highlight
            or highlight is None else 0.6, showlegend=False, hoverinfo="skip"))
    hover = rows["team"] + "<br>Matches: " + rows.get("matches", pd.Series("", index=rows.index)).astype(str)
    if "actual_stage_name" in rows:
        hover = hover + "<br>Reached: " + rows["actual_stage_name"].fillna("").astype(str)
    fig.add_trace(go.Scatter(
        x=rows["expected_points"], y=labels, mode="markers", name="Expected from senior Elo",
        marker={"color": LIGHT_GREY, "size": 12, "line": {"color": GREY, "width": 1}},
        customdata=hover, hovertemplate="%{customdata}<br>Expected: %{x:.1f} points<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=rows["actual_points"], y=labels, mode="markers", name="Actual",
        marker={"color": [ABOVE if v >= 0 else BELOW for v in rows["overperformance"]], "size": 13},
        customdata=hover, hovertemplate="%{customdata}<br>Actual: %{x:.0f} points<extra></extra>"))
    if highlight is not None and highlight in set(rows["team"]):
        row = rows[rows["team"] == highlight].iloc[0]
        fig.add_annotation(
            x=max(row["actual_points"], row["expected_points"]), y=f"<b>{highlight}</b>", xshift=12,
            text=f"{row['actual_points']:.0f} points v {row['expected_points']:.1f} expected "
                 f"({row['overperformance']:+.1f})", showarrow=False, xanchor="left", font={"size": 13})
    fig.update_layout(
        margin={"l": 170, "r": 30, "t": 30, "b": 55},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0},
        yaxis={"categoryorder": "array", "categoryarray": labels[::-1], "title": ""},
        xaxis={"title": "Tournament points (actual against expected)", "rangemode": "tozero"},
    )
    return fig


def expected_vs_actual_points(
    team_table: pd.DataFrame,
    tournament: str,
    label_n: int = 5,
) -> go.Figure:
    """Compare actual with expected tournament points and label the largest residuals."""
    _require_columns(team_table, ["tournament", "team", "expected_points", "actual_points"])
    if not isinstance(label_n, (int, np.integer)) or isinstance(label_n, bool) or label_n < 0:
        raise ValueError("label_n must be a non-negative integer.")
    rows = _tournament_rows(team_table, tournament).copy()
    if rows.empty:
        return _figure()
    rows["_residual"] = rows["actual_points"] - rows["expected_points"]
    labelled = set(rows.nlargest(label_n, "_residual")["team"])
    labelled.update(rows.nsmallest(label_n, "_residual")["team"])
    labels = [team if team in labelled else "" for team in rows["team"]]
    lower = float(min(rows["expected_points"].min(), rows["actual_points"].min()))
    upper = float(max(rows["expected_points"].max(), rows["actual_points"].max()))

    fig = _figure()
    fig.add_trace(
        go.Scatter(
            x=rows["expected_points"],
            y=rows["actual_points"],
            text=labels,
            mode="markers+text",
            textposition="top center",
            marker={"color": ACCENT, "size": 10},
            customdata=rows["team"],
            hovertemplate=(
                "%{customdata}<br>Expected: %{x:.2f} points"
                "<br>Actual: %{y:.2f} points<extra></extra>"
            ),
            name="Teams",
        )
    )
    fig.add_shape(
        type="line",
        x0=lower,
        y0=lower,
        x1=upper,
        y1=upper,
        line={"color": GREY, "dash": "dash"},
    )
    fig.update_layout(
        showlegend=False,
        margin={"l": 75, "r": 20, "t": 15, "b": 65},
        xaxis_title="Expected points (tournament total)",
        yaxis_title="Actual points (tournament total)",
    )
    return fig


def expected_vs_actual_stage(expected: pd.DataFrame, tournament: str) -> go.Figure:
    """Show each team's expected and actual tournament stage on the same ordered scale."""
    required = [
        "tournament", "team", "actual_stage", "actual_stage_name",
        "expected_stage", "progress_over_expectation",
    ]
    _require_columns(expected, required)
    rows = _tournament_rows(expected, tournament).sort_values(
        "progress_over_expectation", ascending=True, kind="stable", ignore_index=True
    )
    if rows.empty:
        return _figure()
    if "p_reach_round_of_32" in rows:
        with_round_of_32 = bool(rows["p_reach_round_of_32"].notna().any())
    else:
        with_round_of_32 = bool(
            rows["actual_stage_name"].astype(str).str.replace("_", " ").str.contains("round of 32", case=False).any()
        )
    stages = STAGES_WITH_ROUND_OF_32 if with_round_of_32 else STAGES_WITHOUT_ROUND_OF_32

    fig = _figure()
    for _, row in rows.iterrows():
        fig.add_shape(
            type="line",
            x0=float(row["expected_stage"]),
            x1=float(row["actual_stage"]),
            y0=row["team"],
            y1=row["team"],
            xref="x",
            yref="y",
            line={"color": LIGHT_GREY, "width": 2},
            layer="below",
        )
    fig.add_trace(
        go.Scatter(
            x=rows["expected_stage"],
            y=rows["team"],
            mode="markers",
            marker={"color": "white", "size": 10, "line": {"color": GREY, "width": 2}},
            hovertemplate="%{y}<br>Expected stage index: %{x:.2f}<extra></extra>",
            name="Expected",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rows["actual_stage"],
            y=rows["team"],
            mode="markers",
            marker={"color": ACCENT, "size": 10},
            customdata=np.column_stack([rows["actual_stage_name"], rows["expected_stage"]]),
            hovertemplate=(
                "%{y}<br>Actual stage: %{customdata[0]}"
                "<br>Expected stage index: %{customdata[1]:.2f}<extra></extra>"
            ),
            name="Actual",
        )
    )
    fig.update_layout(
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0},
        margin={"l": 120, "r": 20, "t": 15, "b": 55},
        xaxis={
            "tickmode": "array",
            "tickvals": list(range(len(stages))),
            "ticktext": stages,
            "range": [-0.2, len(stages) - 0.8],
            "title": "Tournament stage",
        },
        yaxis={"categoryorder": "array", "categoryarray": rows["team"].tolist(), "title": "Team"},
    )
    return fig


def calibration_chart(calibration: pd.DataFrame, tournament: str) -> go.Figure:
    """Plot predicted win probability against observed share for each calibration bin."""
    _require_columns(calibration, ["tournament", "p_win_mean", "win_share"])
    rows = _tournament_rows(calibration, tournament).sort_values(
        "bin" if "bin" in calibration else "p_win_mean", ignore_index=True
    )
    fig = _figure()
    if rows.empty:
        return fig
    fig.add_trace(
        go.Scatter(
            x=rows["p_win_mean"],
            y=rows["win_share"],
            mode="markers",
            marker={"color": ACCENT, "size": 11},
            customdata=rows["bin"] if "bin" in rows else np.arange(1, len(rows) + 1),
            hovertemplate=(
                "Bin %{customdata}<br>Predicted win probability: %{x:.2f}"
                "<br>Observed win share: %{y:.2f}<extra></extra>"
            ),
            name="Calibration bins",
        )
    )
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line={"color": GREY, "dash": "dash"},
    )
    fig.update_layout(
        showlegend=False,
        margin={"l": 75, "r": 20, "t": 15, "b": 65},
        xaxis={"range": [0, 1], "title": "Predicted win probability"},
        yaxis={"range": [0, 1], "title": "Observed share of wins"},
    )
    return fig
