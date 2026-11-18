"""Formation timelines, average team shapes and results by formation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.figure import Figure
from mplsoccer import Pitch

from three_trees.tracking.formations import best_line_count, jenks_labels, formation_string

ACCENT = "#0072B2"
OPPONENT = "#E69F00"
GREY = "#777777"
LIGHT_GREY = "#B8B8B8"
FORMATION_COLOURS = (ACCENT, OPPONENT, GREY, LIGHT_GREY)
POSITION_COLOURS = {
    "Defender": ACCENT,
    "Midfielder": GREY,
    "Forward": OPPONENT,
    "Unknown": LIGHT_GREY,
}


def _figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="simple_white", font={"size": 14})
    return fig


def formation_timeline(formation_team_window: pd.DataFrame, team: str, match_id: int) -> go.Figure:
    """Return a two-row formation timeline with window depths in the hover text."""
    fig = _figure()
    rows = formation_team_window.loc[
        formation_team_window["team"].eq(team) & formation_team_window["match_id"].eq(match_id)
    ].dropna(subset=["formation"])
    if rows.empty:
        return fig
    states = {"with_ball": "With the ball", "without_ball": "Without the ball"}
    formations = list(pd.unique(rows["formation"]))
    colours = {formation: FORMATION_COLOURS[i % len(FORMATION_COLOURS)]
               for i, formation in enumerate(formations)}
    for formation in formations:
        for state, label in states.items():
            part = rows.loc[rows["formation"].eq(formation) & rows["state"].eq(state)]
            if part.empty:
                continue
            start = pd.to_numeric(part["window_start_min"], errors="coerce").to_numpy(float)
            minutes = [f"{value:g}-{value + 5:g} min" for value in start]
            fig.add_trace(go.Bar(
                x=np.full(len(part), 5.0),
                base=start,
                y=[label] * len(part),
                orientation="h",
                name=str(formation),
                legendgroup=str(formation),
                showlegend=state == "with_ball",
                marker={"color": colours[formation], "line": {"color": "white", "width": 1}},
                text=part["formation"],
                textposition="inside",
                customdata=np.column_stack([
                    minutes,
                    pd.to_numeric(part["back_line_m"], errors="coerce"),
                    pd.to_numeric(part["front_line_m"], errors="coerce"),
                ]),
                hovertemplate=(
                    "%{customdata[0]}<br>%{y}<br>Formation: %{text}"
                    "<br>Back line: %{customdata[1]:.1f} m"
                    "<br>Front line: %{customdata[2]:.1f} m<extra></extra>"
                ),
            ))
    fig.add_vline(x=45, line_color=GREY, line_dash="dash")
    fig.update_layout(
        barmode="overlay",
        margin={"l": 120, "r": 20, "t": 20, "b": 55},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0},
        xaxis={"title": "Match minute", "range": [0, 90], "dtick": 5},
        yaxis={"categoryorder": "array", "categoryarray": ["Without the ball", "With the ball"], "title": ""},
    )
    return fig


def average_shape(
    positions_window: pd.DataFrame,
    team: str,
    match_id: int,
    state: str,
    window: tuple[int, int] | float | None = None,
) -> Figure:
    """Return a pitch figure with seconds-weighted average positions and connected depth lines."""
    rows = positions_window.loc[
        positions_window["team"].eq(team)
        & positions_window["match_id"].eq(match_id)
        & positions_window["state"].eq(state)
    ].copy()
    if window is not None:
        if isinstance(window, tuple):
            period, window_number = window
            rows = rows.loc[rows["period"].eq(period) & rows["window"].eq(window_number)]
        else:
            rows = rows.loc[rows["window_start_min"].eq(float(window))]
    rows = rows.dropna(subset=["x_m", "y_m", "seconds"])
    if rows.empty:
        figure, axis = Pitch(
            pitch_type="custom", pitch_length=105, pitch_width=68, line_color=GREY, pitch_color="white",
        ).draw()
        return figure

    rows["weighted_x"] = rows["x_m"] * rows["seconds"]
    rows["weighted_y"] = rows["y_m"] * rows["seconds"]
    players = rows.groupby(["person_id", "position_group"], as_index=False).agg(
        seconds=("seconds", "sum"), weighted_x=("weighted_x", "sum"), weighted_y=("weighted_y", "sum"),
    )
    players["x_m"] = players["weighted_x"] / players["seconds"]
    players["y_m"] = players["weighted_y"] / players["seconds"]
    depths = players["x_m"].to_numpy(dtype=float)
    labels = jenks_labels(depths, best_line_count(depths)) if len(depths) else np.empty(0, dtype=int)
    players["line"] = labels
    figure, axis = Pitch(
        pitch_type="custom", pitch_length=105, pitch_width=68, line_color=GREY, pitch_color="white",
    ).draw()
    for line in np.unique(labels):
        part = players.loc[players["line"].eq(line)].sort_values("y_m", ascending=False)
        if len(part) > 1:
            axis.plot(part["x_m"], part["y_m"], color=LIGHT_GREY, alpha=0.8, linewidth=1.5, zorder=2)
    for group, part in players.groupby("position_group", sort=False):
        axis.scatter(
            part["x_m"], part["y_m"], s=70, color=POSITION_COLOURS.get(group, GREY),
            edgecolors="white", linewidths=0.8, label=group, zorder=3,
        )
    if len(players):
        axis.text(2, 66, formation_string(depths), fontsize=14, fontweight="bold", color=GREY, va="top")
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)
    figure.tight_layout()
    return figure


def _team_shapes(formation_table: pd.DataFrame, team_summary: pd.DataFrame, column: str) -> pd.DataFrame:
    keys = {"tournament", "match_id", "team", column}
    missing = keys - set(formation_table.columns)
    if missing:
        raise ValueError(f"formation table is missing columns: {', '.join(sorted(missing))}.")
    rows = formation_table.dropna(subset=[column]).copy()
    if column == "main_formation" and "state" in rows:
        rows = rows.loc[rows["state"].eq("without_ball")]
    team_keys = ["tournament", "team"]
    if "matches" in team_summary:
        expected = team_summary[team_keys + ["matches"]].drop_duplicates(team_keys)
        coverage = rows.groupby(team_keys, as_index=False)["match_id"].nunique().rename(
            columns={"match_id": "formation_matches"}
        ).merge(expected, on=team_keys, how="right")
        incomplete = coverage["formation_matches"].lt(coverage["matches"]).fillna(True)
        incomplete_teams = coverage.loc[incomplete, team_keys]
        if not incomplete_teams.empty:
            missing_keys = pd.MultiIndex.from_frame(incomplete_teams)
            row_keys = pd.MultiIndex.from_frame(rows[team_keys])
            rows = rows.loc[~row_keys.isin(missing_keys)]
    rows = rows.sort_values("match_id", kind="stable")
    output = []
    for keys, group in rows.groupby(["tournament", "team"], sort=False):
        first_seen = group[column].drop_duplicates().tolist()
        counts = group[column].value_counts()
        output.append({
            "tournament": keys[0],
            "team": keys[1],
            "formation": next(shape for shape in first_seen if counts[shape] == counts.max()),
        })
    return pd.DataFrame(output, columns=["tournament", "team", "formation"])


def formation_results(
    formation_team_match: pd.DataFrame,
    team_summary: pd.DataFrame,
    column: str,
) -> go.Figure:
    """Return team-level mean points above expectation by formation with 95% intervals."""
    if not {"tournament", "team", "overperformance_per_match"}.issubset(team_summary.columns):
        raise ValueError("team_summary must contain tournament, team and overperformance_per_match.")
    shapes = _team_shapes(formation_team_match, team_summary, column)
    rows = shapes.merge(
        team_summary[["tournament", "team", "overperformance_per_match"]],
        on=["tournament", "team"],
        how="inner",
        validate="one_to_one",
    ).dropna(subset=["overperformance_per_match"])
    summaries = []
    for (tournament, formation), group in rows.groupby(["tournament", "formation"], sort=False):
        values = pd.to_numeric(group["overperformance_per_match"], errors="coerce").dropna()
        if len(values) < 2:
            continue
        mean = float(values.mean())
        error = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
        summaries.append({
            "tournament": tournament,
            "formation": formation,
            "mean": mean,
            "low": mean - error,
            "high": mean + error,
            "teams": int(len(values)),
            "team_names": ", ".join(sorted(group.loc[values.index, "team"].astype(str))),
        })
    summary = pd.DataFrame(summaries)
    fig = _figure()
    if summary.empty:
        return fig
    summary = summary.sort_values("mean", kind="stable")
    labels = [f"{t.title()} · {f} (n={n})"
              for t, f, n in zip(summary["tournament"], summary["formation"], summary["teams"])]
    fig.add_trace(go.Scatter(
        x=summary["mean"],
        y=labels,
        mode="markers",
        marker={"color": ACCENT, "size": 12},
        error_x={
            "type": "data",
            "symmetric": False,
            "array": summary["high"] - summary["mean"],
            "arrayminus": summary["mean"] - summary["low"],
            "color": LIGHT_GREY,
            "thickness": 1.5,
        },
        customdata=summary["team_names"],
        hovertemplate="%{y}<br>%{x:+.2f} points per match above expectation"
                      "<br>Teams: %{customdata}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=GREY, line_dash="dash")
    fig.update_layout(
        margin={"l": 210, "r": 20, "t": 20, "b": 60},
        showlegend=False,
        xaxis_title="Mean points above expectation per match (95% interval)",
        yaxis_title="",
    )
    return fig
