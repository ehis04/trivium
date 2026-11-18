"""Plotly figures for tactical profiles (models/tactical_profiles.py): the profile map, how each
profile did against its Elo expectation, and which score goes with beating it.

Returns: team- or profile-level figures with no embedded titles.
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
QUADRANTS = ((1, 1, "Dominant and efficient"), (-1, 1, "Efficient without dominating"),
             (1, -1, "Dominant, not efficient"), (-1, -1, "Neither"))


def _figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="simple_white", font={"size": 14})
    return fig


def profile_map(teams: pd.DataFrame, tournament: str, highlight: str | None = None) -> go.Figure:
    """Teams by dominant (x) and efficient (y) score, coloured by points above Elo expectation per match;
    filled = resourceful, open = costly; the highlighted team labelled."""
    rows = teams[teams["tournament"] == tournament].dropna(subset=["dominant", "efficient"])
    fig = _figure()
    if rows.empty:
        return fig
    limit = float(np.nanmax(np.abs(rows["overperformance_per_match"]))) or 1.0
    hover = (rows["team"] + "<br>" + rows["profile"] + "<br>Points above expectation per match: "
             + rows["overperformance_per_match"].map("{:+.2f}".format)
             + "<br>Reached: " + rows["actual_stage_name"].fillna("").astype(str).str.replace("_", " "))
    for resourceful, symbol, name in ((True, "circle", "Resourceful"), (False, "circle-open", "Costly")):
        part = rows[(rows["resourceful"] >= 0) == resourceful]
        fig.add_trace(go.Scatter(
            x=part["dominant"], y=part["efficient"], mode="markers", name=name,
            marker={"symbol": symbol, "size": 15, "color": part["overperformance_per_match"],
                    "colorscale": [[0, BELOW], [0.5, "#E3E3E3"], [1, ABOVE]], "cmin": -limit, "cmax": limit,
                    "line": {"width": 2}, "showscale": resourceful,
                    "colorbar": {"title": "Points above<br>expectation<br>per match", "thickness": 14}},
            customdata=hover[part.index], hovertemplate="%{customdata}<extra></extra>"))
    if highlight is not None and highlight in set(rows["team"]):
        r = rows[rows["team"] == highlight].iloc[0]
        fig.add_annotation(x=r["dominant"], y=r["efficient"], text=f"<b>{highlight}</b>", showarrow=True,
                           arrowhead=0, ax=40, ay=-35, font={"size": 14, "color": ACCENT})
    span = float(np.nanmax(np.abs(rows[["dominant", "efficient"]].to_numpy()))) + 0.4
    for sx, sy, text in QUADRANTS:
        fig.add_annotation(x=sx * span * 0.98, y=sy * span * 0.98, text=text, showarrow=False,
                           xanchor="right" if sx > 0 else "left", yanchor="top" if sy > 0 else "bottom",
                           font={"color": GREY, "size": 13})
    fig.add_hline(y=0, line_color=LIGHT_GREY)
    fig.add_vline(x=0, line_color=LIGHT_GREY)
    fig.update_layout(
        margin={"l": 60, "r": 20, "t": 30, "b": 60},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0},
        xaxis={"title": "Dominant (standard deviations from the tournament average)", "range": [-span, span],
               "zeroline": False},
        yaxis={"title": "Efficient (standard deviations)", "range": [-span, span], "zeroline": False},
    )
    return fig


def team_scores(teams: pd.DataFrame, tournament: str, team: str) -> go.Figure:
    """Horizontal bar chart of the team's three scores (dominant, efficient, resourceful)
    in standard deviations from the tournament average, with a zero line."""
    fig = _figure()
    if teams.empty or "tournament" not in teams or "team" not in teams:
        return fig
    rows = teams.loc[teams["tournament"].eq(tournament) & teams["team"].eq(team)]
    if rows.empty:
        return fig
    row = rows.iloc[0]
    axes = [("Dominant", "dominant"), ("Efficient", "efficient"), ("Resourceful", "resourceful")]
    labels = [label for label, _ in axes]
    values = [float(row[col]) if col in row and pd.notna(row[col]) else 0.0 for _, col in axes]

    colours = [ABOVE if v >= 0 else BELOW for v in values]
    fig.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": colours},
            hovertemplate="%{y}: %{x:+.2f} SD<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color=LIGHT_GREY, line_width=1.5)
    span = max(abs(v) for v in values) if values else 1.0
    limit = max(span + 0.3, 1.0)
    fig.update_layout(
        margin={"l": 110, "r": 20, "t": 20, "b": 40},
        xaxis={"title": "Standard deviations from tournament average", "range": [-limit, limit], "zeroline": False},
        yaxis={"autorange": "reversed", "title": ""},
        showlegend=False,
    )
    return fig


def profile_outcomes(summary: pd.DataFrame, tournament: str, highlight_profile: str | None = None) -> go.Figure:
    """Per profile: mean points above Elo expectation per match with its 95% interval and the number of teams;
    the highlighted profile in the accent colour."""
    rows = summary[summary["tournament"] == tournament].sort_values("overperformance_per_match")
    fig = _figure()
    if rows.empty:
        return fig
    colours = [ACCENT if p == highlight_profile else (ABOVE if v > 0 else BELOW)
               for p, v in zip(rows["profile"], rows["overperformance_per_match"])]
    labels = [f"{p}  (n={n})" for p, n in zip(rows["profile"], rows["n_teams"])]
    fig.add_trace(go.Scatter(
        x=rows["overperformance_per_match"], y=labels, mode="markers",
        marker={"color": colours, "size": 14},
        error_x={"type": "data", "symmetric": False,
                 "array": (rows["high"] - rows["overperformance_per_match"]).fillna(0),
                 "arrayminus": (rows["overperformance_per_match"] - rows["low"]).fillna(0), "color": LIGHT_GREY,
                 "thickness": 2},
        customdata=rows["teams"],
        hovertemplate="%{y}<br>%{x:+.2f} points per match above expectation<br>%{customdata}<extra></extra>"))
    fig.add_vline(x=0, line_color=GREY, line_dash="dash")
    fig.update_layout(margin={"l": 330, "r": 20, "t": 20, "b": 55}, showlegend=False,
                      xaxis_title="Points above Elo expectation per match (95% interval)", yaxis_title="")
    return fig


def score_associations(axes: pd.DataFrame) -> go.Figure:
    """Per score and tournament: rank correlation with points above expectation per match and its 95%
    interval; filled = repeatable and predicts later matches, open otherwise."""
    fig = _figure()
    if axes.empty:
        return fig
    for tournament, colour, offset in (("men", ACCENT, 0.12), ("women", "#CC79A7", -0.12)):
        rows = axes[axes["tournament"] == tournament]
        if rows.empty:
            continue
        y = [list(axes["label"].drop_duplicates()).index(label) + offset for label in rows["label"]]
        solid = rows["repeatable"].fillna(False).astype(bool) & rows["predicts_later"].fillna(False).astype(bool)
        notes = ["repeatable" if r else "not repeatable" for r in rows["repeatable"].fillna(False)]
        later = ["predicts later matches" if p else "does not clearly predict later matches"
                 for p in rows["predicts_later"].fillna(False)]
        fig.add_trace(go.Scatter(
            x=rows["spearman"], y=y, mode="markers", name=tournament.capitalize(),
            marker={"color": colour, "size": 14, "symbol": ["circle" if s else "circle-open" for s in solid],
                    "line": {"width": 2, "color": colour}},
            error_x={"type": "data", "symmetric": False, "array": rows["spearman_high"] - rows["spearman"],
                     "arrayminus": rows["spearman"] - rows["spearman_low"], "color": colour, "thickness": 1.5},
            customdata=np.column_stack([rows["label"], notes, later, rows["parts"]]),
            hovertemplate="%{customdata[0]} (" + tournament + ")<br>%{x:.2f}<br>%{customdata[1]}, %{customdata[2]}"
                          "<br>%{customdata[3]}<extra></extra>"))
    labels = list(axes["label"].drop_duplicates())
    fig.add_vline(x=0, line_color=GREY, line_dash="dash")
    fig.update_layout(
        margin={"l": 130, "r": 20, "t": 40, "b": 60},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0},
        yaxis={"tickvals": list(range(len(labels))), "ticktext": labels, "autorange": "reversed", "title": ""},
        xaxis={"title": "Rank correlation with points above Elo expectation per match (95% interval)",
               "range": [-1, 1]},
    )
    return fig


def sensitivity_chart(table: pd.DataFrame, tournament: str) -> go.Figure:
    """Per score and definition variant: rank correlation with its 95% team-bootstrap interval."""
    rows = table[table["tournament"] == tournament].copy()
    fig = _figure()
    if rows.empty:
        return fig

    labels = {"dominant": "Dominant", "efficient": "Efficient", "resourceful": "Resourceful"}
    variant_labels = {
        "main": "main definition",
        "contests_share": "contests alone",
        "territory_share": "territory alone",
        "chances_share": "chances alone",
        "conversion": "conversion alone",
        "security": "security alone",
        "defensive_work_per_dominance": "defensive work alone",
        "hsr_without_ball_per_dominance": "high-speed running alone",
        "group_stage_only": "group matches only",
        "without_strongest": "without strongest quarter",
    }
    score_order = ["dominant", "efficient", "resourceful"]
    score_colours = {"dominant": ACCENT, "efficient": ABOVE, "resourceful": "#CC79A7"}
    for score in score_order:
        part = rows[rows["score"] == score]
        if part.empty:
            continue
        part = part.sort_values("variant", key=lambda s: s.map(
            {name: index for index, name in enumerate(variant_labels)}))
        y = [f"{labels.get(score, score)} — {variant_labels.get(variant, variant)}"
             for variant in part["variant"]]
        main = part["variant"].eq("main")
        fig.add_trace(go.Scatter(
            x=part["spearman"], y=y, mode="markers", name=labels.get(score, score),
            marker={"color": [ACCENT if is_main else score_colours[score] for is_main in main],
                    "symbol": ["diamond" if is_main else "circle" for is_main in main],
                    "size": [15 if is_main else 11 for is_main in main],
                    "line": {"width": 1.5, "color": score_colours[score]}},
            error_x={"type": "data", "symmetric": False,
                     "array": (part["high"] - part["spearman"]).fillna(0).clip(lower=0),
                     "arrayminus": (part["spearman"] - part["low"]).fillna(0).clip(lower=0),
                     "color": score_colours[score], "thickness": 1.5},
            customdata=np.column_stack([
                part["n_teams"], part["variant"].map(variant_labels).fillna(part["variant"]),
                part["same_sign_as_main"].fillna(False),
            ]),
            hovertemplate=("%{customdata[1]}<br>Spearman: %{x:.2f} "
                           "(95% interval)<br>Teams: %{customdata[0]}"
                           "<br>Same sign as main: %{customdata[2]}<extra></extra>"),
        ))
    fig.add_vline(x=0, line_color=GREY, line_dash="dash")
    fig.update_layout(
        margin={"l": 300, "r": 20, "t": 20, "b": 55},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0},
        xaxis={"title": "Rank correlation with points above Elo expectation per match (95% interval)",
               "range": [-1, 1]},
        yaxis={"title": "", "autorange": "reversed"},
    )
    return fig
