"""Plotly figures for team-level running, FIFA phase and defensive shape."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from three_trees.measures import available, measures

ACCENT = "#0072B2"
OPPONENT = "#E69F00"
GREY = "#777777"
LIGHT_GREY = "#B8B8B8"

PHASE_ORDER = (
    "build_up",
    "progression",
    "final_third",
    "long_ball",
    "attacking_transition",
    "counter_attack",
    "press",
    "counter_press",
    "defensive_transition",
    "recovery",
    "block",
)
SHAPE_MEASURES = ("line_height_m", "team_length_m", "team_width_m")
POSITION_GROUPS = ("Defender", "Midfielder", "Forward")
STATES = ("with_ball", "without_ball", "out_of_play")
THIRDS = ("defensive", "middle", "attacking")
THIRD_LABELS = {"defensive": "Own third", "middle": "Middle third", "attacking": "Final third"}


def _figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="simple_white", font={"size": 14})
    return fig


def _registry(table: pd.DataFrame) -> pd.DataFrame:
    return available(table, measures(tournament="men", table="team_match"))


def _display_label(row: pd.Series) -> str:
    label = str(row["label"])
    unit = str(row["unit"])
    return f"{label} ({unit})" if unit and unit != "share" else label


def cost_of_dominance(team_tournament: pd.DataFrame, team: str) -> go.Figure:
    """Compare each team's mean dominance with its high-speed running cost per match."""
    fig = _figure()
    required = {"team", "matches", "dominance", "hsr_without_ball_km"}
    registry = _registry(team_tournament).set_index("column")
    if not required.issubset(team_tournament.columns) or not {
        "dominance", "hsr_without_ball_km"
    }.issubset(registry.index):
        return fig

    rows = team_tournament.copy()
    for column in ("dominance", "hsr_without_ball_km", "matches"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = rows.dropna(subset=["team", "dominance", "hsr_without_ball_km"])
    if rows.empty:
        return fig

    x_median = float(rows["dominance"].median() * 100)
    y_median = float(rows["hsr_without_ball_km"].median())
    dominance_label = _display_label(registry.loc["dominance"])
    cost_label = f"{_display_label(registry.loc['hsr_without_ball_km'])} per match"
    for selected, color, mode in ((False, GREY, "markers"), (True, ACCENT, "markers+text")):
        side = rows.loc[rows["team"].eq(team)] if selected else rows.loc[~rows["team"].eq(team)]
        if side.empty:
            continue
        fig.add_trace(go.Scatter(
            x=side["dominance"] * 100,
            y=side["hsr_without_ball_km"],
            mode=mode,
            name=team if selected else "Other teams",
            marker={"color": color, "size": 11},
            text=side["team"] if selected else None,
            textposition="top right",
            customdata=side[["team", "matches"]].to_numpy(),
            hovertemplate=(
                "%{customdata[0]}<br>Matches: %{customdata[1]:.0f}<br>"
                f"{dominance_label}: %{{x:.1f}}%<br>{cost_label}: %{{y:.2f}}"
                "<extra></extra>"
            ),
        ))

    fig.add_vline(x=x_median, line_color=GREY, line_dash="dash")
    fig.add_hline(y=y_median, line_color=GREY, line_dash="dash")
    fig.update_layout(
        margin={"l": 60, "r": 30, "t": 50, "b": 60},
        xaxis={"title": f"{dominance_label} (%)"},
        yaxis={"title": cost_label},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        annotations=[
            {"x": 0.02, "y": 0.02, "xref": "paper", "yref": "paper", "text": "Less dominant, low cost",
             "showarrow": False, "xanchor": "left", "yanchor": "bottom", "font": {"color": GREY, "size": 12}},
            {"x": 0.98, "y": 0.02, "xref": "paper", "yref": "paper", "text": "Dominant at low cost",
             "showarrow": False, "xanchor": "right", "yanchor": "bottom", "font": {"color": GREY, "size": 12}},
            {"x": 0.02, "y": 0.98, "xref": "paper", "yref": "paper", "text": "Chasing the game",
             "showarrow": False, "xanchor": "left", "yanchor": "top", "font": {"color": GREY, "size": 12}},
            {"x": 0.98, "y": 0.98, "xref": "paper", "yref": "paper", "text": "Dominant at high cost",
             "showarrow": False, "xanchor": "right", "yanchor": "top", "font": {"color": GREY, "size": 12}},
        ],
    )
    return fig


def running_heatmap(running: pd.DataFrame, team: str) -> go.Figure:
    """Show team high-speed running per 90 minutes by position, possession state and pitch third."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=POSITION_GROUPS, shared_yaxes=True)
    fig.update_layout(template="simple_white", font={"size": 14})
    required = {"team", "position_group", "state", "third", "hsr_m", "minutes"}
    if not required.issubset(running.columns) or running.empty:
        return fig

    rows = running.loc[
        running["position_group"].isin(POSITION_GROUPS)
        & running["state"].isin(STATES)
        & running["third"].isin(THIRDS)
    ].copy()
    rows["hsr_m"] = pd.to_numeric(rows["hsr_m"], errors="coerce")
    rows["minutes"] = pd.to_numeric(rows["minutes"], errors="coerce")
    rows = rows.dropna(subset=["team", "hsr_m", "minutes"])
    if rows.empty or not rows["team"].eq(team).any():
        return fig

    team_cells = rows.groupby(
        ["team", "position_group", "state", "third"], as_index=False
    )[["hsr_m", "minutes"]].sum()
    team_cells["rate"] = team_cells["hsr_m"] / team_cells["minutes"].where(team_cells["minutes"] > 0) * 90
    tournament_cells = team_cells.groupby(
        ["position_group", "state", "third"], as_index=False
    )["rate"].mean()
    selected_cells = team_cells.loc[team_cells["team"].eq(team)]
    running_label = str(measures(tournament="men", table="team_match").set_index("measure").loc["hsr_km", "label"])

    for col, group in enumerate(POSITION_GROUPS, start=1):
        selected_group = selected_cells.loc[selected_cells["position_group"].eq(group)]
        average_group = tournament_cells.loc[tournament_cells["position_group"].eq(group)]
        selected_matrix = selected_group.pivot(index="state", columns="third", values="rate")
        average_matrix = average_group.pivot(index="state", columns="third", values="rate")
        z = selected_matrix.reindex(index=STATES, columns=THIRDS).to_numpy(dtype=float)
        average = average_matrix.reindex(index=STATES, columns=THIRDS).to_numpy(dtype=float)
        text = np.where(np.isfinite(z), np.char.mod("%.0f", z), "")
        fig.add_trace(go.Heatmap(
            z=z,
            x=[THIRD_LABELS[third] for third in THIRDS],
            y=["With ball", "Without ball", "Out of play"],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 13},
            customdata=average,
            colorscale=[[0, "#FFFFFF"], [1, ACCENT]],
            showscale=col == 3,
            colorbar={"title": {"text": f"{running_label} (m/90 min)"}} if col == 3 else None,
            hovertemplate=(
                f"{team}<br>%{{y}} · %{{x}}<br>{running_label}: %{{z:.0f}} m/90 min"
                "<br>Tournament average: %{customdata:.0f} m/90 min<extra></extra>"
            ),
        ), row=1, col=col)
        fig.update_xaxes(title_text="Pitch third", row=1, col=col)
        fig.update_yaxes(title_text="Possession state" if col == 1 else "", row=1, col=col)

    fig.update_layout(
        height=390,
        margin={"l": 100, "r": 30, "t": 45, "b": 60},
    )
    return fig


def fifa_phase_profile(team_match_provider: pd.DataFrame, team: str) -> go.Figure:
    """Compare the team's mean share of each FIFA phase with the tournament average."""
    fig = _figure()
    if "team" not in team_match_provider.columns:
        return fig
    registry = _registry(team_match_provider).set_index("column")
    columns = [
        f"fifa_{phase}_share" for phase in PHASE_ORDER
        if f"fifa_{phase}_share" in registry.index and f"fifa_{phase}_share" in team_match_provider.columns
    ]
    if not columns:
        return fig

    selected = team_match_provider.loc[team_match_provider["team"].eq(team)]
    if selected.empty:
        return fig
    labels = [str(registry.loc[column, "label"]) for column in columns]
    team_values = [pd.to_numeric(selected[column], errors="coerce").mean() * 100 for column in columns]
    tournament_values = [pd.to_numeric(team_match_provider[column], errors="coerce").mean() * 100 for column in columns]
    custom = [[team_values[i], tournament_values[i]] for i in range(len(columns))]
    fig.add_trace(go.Scatter(
        x=tournament_values,
        y=labels,
        mode="markers",
        name="Tournament average",
        marker={"color": GREY, "size": 10},
        customdata=custom,
        hovertemplate="%{y}<br>Tournament average: %{x:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=team_values,
        y=labels,
        mode="markers",
        name=team,
        marker={"color": ACCENT, "size": 11},
        customdata=custom,
        hovertemplate=f"{team}<br>%{{y}}: %{{x:.1f}}%<br>Tournament average: %{{customdata[1]:.1f}}%<extra></extra>",
    ))
    fig.update_layout(
        margin={"l": 210, "r": 30, "t": 25, "b": 50},
        xaxis={"title": "Share of phase time (%)", "ticksuffix": "%"},
        yaxis={"title": "", "autorange": "reversed"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    return fig


def shape_profile(team_match_provider: pd.DataFrame, team: str) -> go.Figure:
    """Compare the team's defensive shape with tournament means and team-level ranges."""
    fig = _figure()
    if "team" not in team_match_provider.columns:
        return fig
    registry = _registry(team_match_provider).set_index("column")
    columns = [column for column in SHAPE_MEASURES if column in registry.index and column in team_match_provider]
    if not columns:
        return fig
    selected = team_match_provider.loc[team_match_provider["team"].eq(team)]
    if selected.empty:
        return fig

    team_means = team_match_provider.groupby("team")[columns].mean()
    selected_means = selected[columns].mean()
    labels = [_display_label(registry.loc[column]) for column in columns]
    ranges = []
    averages = team_means[columns].mean().tolist()
    selected_values = []
    for column in columns:
        values = pd.to_numeric(team_means[column], errors="coerce").dropna()
        ranges.extend([float(values.min()), float(values.max()), None] if not values.empty else [None, None, None])
        selected_values.append(float(selected_means[column]))

    fig.add_trace(go.Scatter(
        x=ranges,
        y=[item for label in labels for item in (label, label, None)],
        mode="lines",
        line={"color": LIGHT_GREY, "width": 10},
        name="Range across teams",
        showlegend=False,
        hoverinfo="skip",
        connectgaps=False,
    ))
    fig.add_trace(go.Scatter(
        x=averages,
        y=labels,
        mode="markers",
        name="Tournament average",
        marker={"color": GREY, "size": 10},
        customdata=[str(registry.loc[column, "unit"]) for column in columns],
        hovertemplate="%{y}<br>Tournament average: %{x:.1f} %{customdata}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=selected_values,
        y=labels,
        mode="markers",
        name=team,
        marker={"color": ACCENT, "size": 11},
        customdata=[str(registry.loc[column, "unit"]) for column in columns],
        hovertemplate=f"{team}<br>%{{y}}: %{{x:.1f}} %{{customdata}}<extra></extra>",
    ))
    fig.update_layout(
        margin={"l": 250, "r": 30, "t": 25, "b": 50},
        xaxis={"title": str(registry.loc[columns[0], "unit"]).title()},
        yaxis={"title": "", "autorange": "reversed"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    return fig
