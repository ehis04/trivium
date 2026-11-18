"""Simple charts that answer one question at a glance, shown next to the fuller charts that carry the intervals:
teams ranked on one bar chart, teams as dots with a straight line of best fit, bars per profile, bars per measure,
a team against the tournament average, team against opponent in a match, and a pitch split into thirds.

Returns: Plotly figures (matplotlib for the pitch) with no embedded titles; team or profile level only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mplsoccer import Pitch

from three_trees.charts.pitch import MEASURES, SHARE_CMAP, SHARE_NORM

ACCENT = "#0072B2"
OPPONENT = "#E69F00"
GREY = "#777777"
LIGHT_GREY = "#B8B8B8"
ABOVE = "#009E73"
BELOW = "#D55E00"
THIRDS = (("Own third", (0, 1)), ("Middle third", (2, 3)), ("Final third", (4, 5)))


def _figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="simple_white", font={"size": 14})
    return fig


def ranked_bars(table: pd.DataFrame, value: str, label: str, highlight: str | None = None) -> go.Figure:
    """One horizontal bar per team, highest at the top; the highlighted team in the accent colour, the rest grey;
    a line at zero when the values cross it, else at the average. table: team and value columns."""
    rows = table.dropna(subset=[value]).sort_values(value, kind="stable")
    fig = _figure()
    if rows.empty:
        return fig
    colours = [ACCENT if team == highlight else LIGHT_GREY for team in rows["team"]]
    fig.add_trace(go.Bar(x=rows[value], y=rows["team"], orientation="h", marker_color=colours,
                         hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
    crosses = rows[value].min() < 0 < rows[value].max()
    reference = 0.0 if crosses else float(rows[value].mean())
    fig.add_vline(x=reference, line_color=GREY, line_dash="dash",
                  annotation_text="" if crosses else "Average", annotation_position="top")
    fig.update_layout(showlegend=False, height=max(350, 16 * len(rows) + 90),
                      margin={"l": 160, "r": 20, "t": 30, "b": 50}, xaxis_title=label, yaxis_title="",
                      yaxis={"tickfont": {"size": 11}})
    return fig


def best_fit(x: pd.Series, y: pd.Series) -> dict[str, float]:
    """Least-squares straight line through the points: slope, intercept, Pearson r and n."""
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(data) < 3 or data["x"].nunique() < 2:
        return {"slope": np.nan, "intercept": np.nan, "r": np.nan, "n": float(len(data))}
    slope, intercept = np.polyfit(data["x"], data["y"], 1)
    return {"slope": float(slope), "intercept": float(intercept), "r": float(data["x"].corr(data["y"])),
            "n": float(len(data))}


def scatter_fit(table: pd.DataFrame, x: str, y: str, x_label: str, y_label: str,
                highlight: str | None = None) -> tuple[go.Figure, dict[str, float]]:
    """Teams as dots with the straight line of best fit; the highlighted team labelled. Returns the figure and the
    fit (slope, intercept, r, n) for the caption. table: team, x and y columns."""
    rows = table.dropna(subset=[x, y])
    fit = best_fit(rows[x], rows[y])
    fig = _figure()
    if rows.empty:
        return fig, fit
    fig.add_trace(go.Scatter(x=rows[x], y=rows[y], mode="markers", name="Teams",
                             marker={"color": [ACCENT if t == highlight else LIGHT_GREY for t in rows["team"]],
                                     "size": 11, "line": {"color": GREY, "width": 1}},
                             customdata=rows["team"], hovertemplate="%{customdata}<br>%{x:.2f}, %{y:.2f}<extra></extra>"))
    if np.isfinite(fit["slope"]):
        xs = np.linspace(rows[x].min(), rows[x].max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=fit["intercept"] + fit["slope"] * xs, mode="lines", name="Line of best fit",
                                 line={"color": ABOVE if fit["slope"] > 0 else BELOW, "width": 3}, hoverinfo="skip"))
    if highlight is not None and highlight in set(rows["team"]):
        r = rows[rows["team"] == highlight].iloc[0]
        fig.add_annotation(x=r[x], y=r[y], text=f"<b>{highlight}</b>", showarrow=True, arrowhead=0, ax=30, ay=-30,
                           font={"color": ACCENT})
    fig.update_layout(showlegend=True, legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0},
                      margin={"l": 60, "r": 20, "t": 40, "b": 60}, xaxis_title=x_label, yaxis_title=y_label)
    return fig, fit


def profile_bars(summary: pd.DataFrame, tournament: str, highlight_profile: str | None = None) -> go.Figure:
    """One bar per profile: the mean points above Elo expectation per match of its teams (green above zero,
    orange below; the highlighted profile outlined), with the number of teams on the bar."""
    rows = summary[summary["tournament"] == tournament].sort_values("overperformance_per_match")
    fig = _figure()
    if rows.empty:
        return fig
    fig.add_trace(go.Bar(
        x=rows["overperformance_per_match"], y=rows["profile"], orientation="h",
        marker={"color": [ABOVE if v > 0 else BELOW for v in rows["overperformance_per_match"]],
                "line": {"color": ["#1B1B1B" if p == highlight_profile else "rgba(0,0,0,0)" for p in rows["profile"]],
                         "width": 3}},
        text=[f"{n} teams" for n in rows["n_teams"]], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>%{x:+.2f} points per match<extra></extra>"))
    fig.add_vline(x=0, line_color=GREY)
    fig.update_layout(showlegend=False, margin={"l": 300, "r": 60, "t": 20, "b": 55},
                      xaxis_title="Points above Elo expectation per match (average of the teams)", yaxis_title="")
    return fig


def evidence_bars(evidence: pd.DataFrame, tournament: str, point: str | None = None) -> go.Figure:
    """One bar per measure (models.answer.point_evidence rows): teams above minus below their Elo expectation in
    standard deviations; coloured when the difference is clear (green higher, orange lower), grey when not."""
    rows = evidence[evidence["tournament"] == tournament]
    if point is not None:
        rows = rows[rows["point"] == point]
    rows = rows.iloc[::-1]
    fig = _figure()
    if rows.empty:
        return fig
    colours = [(ABOVE if v > 0 else BELOW) if clear else LIGHT_GREY
               for v, clear in zip(rows["adjusted_difference_z"], rows["adjusted_excludes_zero"])]
    fig.add_trace(go.Bar(x=rows["adjusted_difference_z"], y=rows["label"], orientation="h", marker_color=colours,
                         customdata=np.where(rows["adjusted_excludes_zero"], "clear", "no clear difference"),
                         hovertemplate="%{y}<br>%{x:+.2f} standard deviations (%{customdata})<extra></extra>"))
    fig.add_vline(x=0, line_color=GREY)
    fig.update_layout(showlegend=False, height=max(300, 34 * len(rows) + 90),
                      margin={"l": 330, "r": 20, "t": 20, "b": 55},
                      xaxis_title="Teams that beat their strength minus the rest (standard deviations)", yaxis_title="")
    return fig


def team_vs_average(team_values: pd.Series, averages: pd.Series, labels: dict[str, str]) -> go.Figure:
    """For each measure, the team's value as a percentage of the tournament average (100 = average): one bar per
    measure, green above the average, orange below. team_values and averages are indexed by column."""
    columns = [c for c in labels if c in team_values.index and pd.notna(team_values[c]) and averages.get(c)]
    fig = _figure()
    if not columns:
        return fig
    percent = [100 * float(team_values[c]) / float(averages[c]) for c in columns]
    fig.add_trace(go.Bar(x=percent, y=[labels[c] for c in columns], orientation="h",
                         marker_color=[ABOVE if p >= 100 else BELOW for p in percent],
                         customdata=np.column_stack([[float(team_values[c]) for c in columns],
                                                     [float(averages[c]) for c in columns]]),
                         hovertemplate="%{y}<br>Team %{customdata[0]:.2f}, average %{customdata[1]:.2f}<extra></extra>",
                         base=0))
    fig.add_vline(x=100, line_color=GREY, line_dash="dash", annotation_text="Tournament average",
                  annotation_position="top")
    fig.update_layout(showlegend=False, margin={"l": 260, "r": 20, "t": 30, "b": 55},
                      xaxis_title="Team as a percentage of the tournament average (100 = average)", yaxis_title="")
    return fig


def match_bars(rows: pd.DataFrame, team: str, opponent: str, labels: dict[str, str]) -> go.Figure:
    """Team against opponent in one match: grouped horizontal bars, one pair per measure in its own units.
    rows: the match's two team_match rows."""
    by_team = rows.set_index("team")
    columns = [c for c in labels if c in by_team.columns]
    fig = _figure()
    if team not in by_team.index or not columns:
        return fig
    for name, colour in ((team, ACCENT), (opponent, OPPONENT)):
        if name not in by_team.index:
            continue
        values = [float(by_team.loc[name, c]) for c in columns]
        fig.add_trace(go.Bar(x=values, y=[labels[c] for c in columns], orientation="h", name=name,
                             marker_color=colour, text=[f"{v:.2f}" if v < 10 else f"{v:.0f}" for v in values],
                             textposition="outside", cliponaxis=False,
                             hovertemplate=f"{name}<br>%{{y}}: %{{x:.2f}}<extra></extra>"))
    fig.update_layout(barmode="group", margin={"l": 200, "r": 40, "t": 30, "b": 40},
                      legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0},
                      yaxis={"autorange": "reversed", "title": ""}, xaxis_title="")
    return fig


def thirds_shares(zones: pd.DataFrame, measure: str = "territory_share") -> pd.Series:
    """The team's share of the counted actions (territory) or contests in its own, middle and final third, from
    team_match_zone or team_tournament_zone rows of one team."""
    won, lost = MEASURES[measure]
    shares = {}
    for name, (first, last) in THIRDS:
        part = zones[zones["zone_x"].between(first, last)]
        total = float(part[won].sum() + part[lost].sum())
        shares[name] = float(part[won].sum()) / total if total else np.nan
    return pd.Series(shares)


def thirds_pitch(zones: pd.DataFrame, measure: str = "territory_share", title: str = "") -> Figure:
    """A pitch split into the team's own, middle and final third (attacking left to right), each third shaded by
    the team's share there (blue: the team ahead, orange: the opponent ahead) and labelled with the share."""
    shares = thirds_shares(zones, measure)
    pitch = Pitch(pitch_type="statsbomb", pitch_color="white", line_color="#7b8794", linewidth=1, line_zorder=2)
    figure, axis = pitch.draw(figsize=(8, 5.5))
    for (name, _), (left, share) in zip(THIRDS, zip((0, 40, 80), shares)):
        if np.isfinite(share):
            axis.add_patch(Rectangle((left, 0), 40, 80, facecolor=SHARE_CMAP(SHARE_NORM(share)), edgecolor="white",
                                     zorder=1))
        axis.text(left + 20, 40, f"{name}\n{share:.0%}" if np.isfinite(share) else f"{name}\nno data",
                  ha="center", va="center", fontsize=15, zorder=3)
    axis.annotate("", xy=(80, 84), xytext=(40, 84), annotation_clip=False,
                  arrowprops={"arrowstyle": "->", "color": "#555555"})
    axis.text(60, 86, "Attacking direction", ha="center", va="top", fontsize=11, color="#555555")
    if title:
        axis.set_title(title, fontsize=14)
    return figure
