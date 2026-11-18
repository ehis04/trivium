"""Plotly figures for one match: the dominance parts against 50% and the cumulative xG timeline.

Returns: figures for the selected team against its opponent, team level only, no embedded titles.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

ACCENT = "#0072B2"
OPPONENT = "#E69F00"
GREY = "#777777"
LIGHT_GREY = "#B8B8B8"
DOMINANCE_PARTS = (
    ("contests_share", "Contests won"),
    ("territory_share", "Field tilt"),
    ("chances_share", "Chances share"),
    ("dominance", "Overall dominance"),
)


def _figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="simple_white", font={"size": 14})
    return fig


def dominance_bars(team_match: pd.DataFrame, team: str) -> go.Figure:
    """The team's share of each dominance part in one match as a bar from 50%: right of 50 = the team
    ahead. team_match: the match's two rows (team plus the share columns)."""
    rows = team_match.set_index("team")
    parts = [(column, label) for column, label in DOMINANCE_PARTS if column in rows]
    fig = _figure()
    if team not in rows.index or not parts:
        return fig
    opponent = next((t for t in rows.index if t != team), "Opponent")
    shares = [100 * float(rows.loc[team, column]) for column, _ in parts]
    labels = [label for _, label in parts]
    fig.add_trace(go.Bar(
        x=[share - 50 for share in shares], y=labels, base=50, orientation="h",
        marker_color=[ACCENT if share >= 50 else OPPONENT for share in shares],
        text=[f"{share:.0f}% v {100 - share:.0f}%" for share in shares], textposition="outside",
        cliponaxis=False,
        customdata=[[share, 100 - share] for share in shares],
        hovertemplate=f"%{{y}}<br>{team}: %{{customdata[0]:.1f}}%<br>{opponent}: %{{customdata[1]:.1f}}%"
                      "<extra></extra>",
    ))
    fig.add_vline(x=50, line_color=GREY, line_dash="dash")
    fig.update_layout(
        showlegend=False,
        margin={"l": 10, "r": 30, "t": 30, "b": 50},
        xaxis={"range": [0, 100], "tickvals": [0, 25, 50, 75, 100],
               "ticktext": ["0%", "25%", "50%", "75%", "100%"], "title": f"{team}'s share (50% = even)"},
        yaxis={"autorange": "reversed", "title": ""},
        annotations=[
            {"x": 2, "y": 1.06, "xref": "x", "yref": "paper", "text": f"← {opponent} ahead", "showarrow": False,
             "xanchor": "left", "font": {"color": OPPONENT}},
            {"x": 98, "y": 1.06, "xref": "x", "yref": "paper", "text": f"{team} ahead →", "showarrow": False,
             "xanchor": "right", "font": {"color": ACCENT}},
        ],
    )
    return fig


def _goal_label(row: pd.Series) -> str:
    tag = " (og)" if row["own_goal"] else " (pen)" if row["penalty"] else ""
    return f"{int(row['minute']) + 1}'{tag}"


def xg_timeline(shots: pd.DataFrame, team: str, opponent: str) -> go.Figure:
    """Cumulative xG of both teams over elapsed playing time (a step at each shot), goals marked with
    the match-clock minute, half-time line. shots: the match's shot_timeline rows."""
    fig = _figure()
    end = max(90.0, float(shots["elapsed_minute"].max()) if len(shots) else 90.0)
    for name, colour in ((team, ACCENT), (opponent, OPPONENT)):
        side = shots[shots["team"] == name].sort_values("elapsed_minute", kind="stable")
        cumulative = side["xg"].cumsum()
        total = float(side["xg"].sum())
        goals = int(side["goal"].sum())
        fig.add_trace(go.Scatter(
            x=[0.0, *side["elapsed_minute"], end], y=[0.0, *cumulative, total], mode="lines",
            line={"shape": "hv", "color": colour, "width": 3},
            name=f"{name}: {total:.2f} xG, {goals} goal{'s' if goals != 1 else ''}",
            hovertemplate=f"{name}<br>%{{x:.0f}} min: %{{y:.2f}} xG<extra></extra>",
        ))
        scored = side[side["goal"]]
        if len(scored):
            fig.add_trace(go.Scatter(
                x=scored["elapsed_minute"], y=cumulative[scored.index], mode="markers+text",
                marker={"size": 13, "color": colour, "line": {"color": "white", "width": 2}},
                text=[_goal_label(row) for _, row in scored.iterrows()], textposition="top left",
                textfont={"color": colour}, showlegend=False,
                hovertemplate=f"{name} goal<br>%{{text}}<extra></extra>",
            ))
    half = shots["half_time_minute"].dropna()
    if len(half):
        fig.add_vline(x=float(half.iloc[0]), line_color=LIGHT_GREY, line_dash="dot",
                      annotation_text="Half-time", annotation_position="top")
    fig.update_layout(
        margin={"l": 60, "r": 20, "t": 40, "b": 50},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        xaxis={"title": "Minutes played", "range": [0, end + 1]},
        yaxis={"title": "Expected goals (xG), cumulative", "rangemode": "tozero"},
    )
    return fig
