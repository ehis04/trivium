"""Plotly charts for team-level trends and phase-level comparisons."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

ACCENT = "#0072B2"
GREY = "#777777"
LIGHT_GREY = "#B8B8B8"
PHASE_ORDER = ["settled", "transition", "counter_attack", "set_piece", "other"]
PHASE_LABELS = {
    "settled": "Settled",
    "transition": "Transition",
    "counter_attack": "Counter-attack",
    "set_piece": "Set piece",
    "other": "Other",
}


def _figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="simple_white", font={"size": 14})
    return fig


def _stage_rows(stage_bounds: object) -> list[dict[str, object]]:
    if stage_bounds is None:
        return []
    if isinstance(stage_bounds, dict):
        return [{"match_number": key, "label": value} for key, value in stage_bounds.items()]
    if isinstance(stage_bounds, pd.DataFrame):
        rows = stage_bounds.copy()
        if "match_number" in rows:
            if "stage" in rows:
                rows = rows.rename(columns={"stage": "label"})
            elif "label" in rows:
                pass
            elif "competition_stage" in rows:
                rows = rows.rename(columns={"competition_stage": "label"})
        if "label" in rows:
            return [{"match_number": int(row["match_number"]), "label": str(row["label"])} for _, row in rows.iterrows()]
    if isinstance(stage_bounds, (list, tuple)):
        return [{"match_number": int(item), "label": "Stage"} for item in stage_bounds]
    return []


def tournament_lines(rows: pd.DataFrame, measure_column: str, label: str, team: str, stage_bounds: object) -> go.Figure:
    """Show each team's line across the tournament, with stage boundaries and the selected team highlighted."""
    required = {"team", "match_number", measure_column}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}.")
    frame = rows[["team", "match_number", measure_column]].dropna(subset=["team", "match_number", measure_column]).copy()
    frame["match_number"] = pd.to_numeric(frame["match_number"], errors="coerce")
    frame[measure_column] = pd.to_numeric(frame[measure_column], errors="coerce")
    frame = frame.dropna(subset=["match_number", measure_column]).sort_values(["team", "match_number"], kind="stable")
    if frame.empty:
        return _figure()
    teams = [current for current in sorted(frame["team"].drop_duplicates().tolist()) if current != team]
    if team in frame["team"].drop_duplicates().tolist():
        teams.append(team)
    fig = _figure()
    for current in teams:
        subset = frame[frame["team"].eq(current)].sort_values("match_number", kind="stable")
        x = subset["match_number"].astype(float).tolist()
        y = subset[measure_column].astype(float).tolist()
        if current == team:
            text = ["" for _ in x]
            last_value = float(subset.iloc[-1][measure_column])
            text[-1] = f"{last_value:.2f}"
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers+text",
                    line={"color": ACCENT, "width": 3},
                    marker={"color": ACCENT, "size": 8},
                    text=text,
                    textposition="top center",
                    textfont={"color": ACCENT, "size": 12},
                    name=current,
                    hovertemplate=f"{current}<br>{label}: %{{y:.2f}}<extra></extra>",
                    showlegend=False,
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line={"color": LIGHT_GREY, "width": 1.5},
                    name=current,
                    hovertemplate=f"{current}<br>{label}: %{{y:.2f}}<extra></extra>",
                    showlegend=False,
                )
            )
    for boundary in _stage_rows(stage_bounds):
        value = float(boundary["match_number"]) - 0.5  # the line sits before the stage's first match
        fig.add_vline(
            x=value,
            line_color=GREY,
            line_dash="dot",
            annotation_text=str(boundary.get("label", "Stage")),
            annotation_position="top left",
        )
    fig.update_layout(
        xaxis_title="Match number",
        yaxis_title=label,
        margin={"l": 60, "r": 20, "t": 30, "b": 45},
        hovermode="closest",
    )
    return fig


def phase_dot_plot(team_phase: pd.DataFrame, tournament_phase: pd.DataFrame, team: str, column: str = "xg_per_possession") -> go.Figure:
    """Compare a team's phase profile with the tournament mean for each phase."""
    if column not in {"xg_per_possession", "share_of_possessions"}:
        raise ValueError("column must be 'xg_per_possession' or 'share_of_possessions'.")
    fig = _figure()
    if team_phase.empty or tournament_phase.empty:
        return fig
    team_rows = team_phase[team_phase["team"].eq(team)].copy() if "team" in team_phase else team_phase.copy()
    average_rows = tournament_phase.copy()
    if "phase" not in team_rows or "phase" not in average_rows:
        return fig
    team_rows["phase"] = team_rows["phase"].astype(str).str.strip().str.lower()
    average_rows["phase"] = average_rows["phase"].astype(str).str.strip().str.lower()
    labels = []
    team_values = []
    average_values = []
    hover_rows = []
    for phase_key in PHASE_ORDER:
        team_row = team_rows[team_rows["phase"].eq(phase_key)]
        if team_row.empty:
            continue
        row = team_row.iloc[0]
        average = average_rows[average_rows["phase"].eq(phase_key)]
        if average.empty:
            continue
        value = pd.to_numeric(row.get(column, pd.NA), errors="coerce")
        avg_value = pd.to_numeric(average[column], errors="coerce").mean()
        if pd.isna(value) or pd.isna(avg_value):
            continue
        labels.append(PHASE_LABELS[phase_key])
        team_values.append(float(value))
        average_values.append(float(avg_value))
        hover_rows.append((row.get("possessions", 0), row.get("shots", 0), row.get("xg", 0)))
    if not labels:
        return fig
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=average_values,
            mode="markers",
            marker={"color": GREY, "size": 10},
            name="Tournament average",
            hovertemplate="Tournament average<br>%{x}<br>%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=team_values,
            mode="markers",
            marker={"color": ACCENT, "size": 11},
            name=team,
            customdata=hover_rows,
            hovertemplate="%{x}<br>Possessions: %{customdata[0]}<br>Shots: %{customdata[1]}<br>xG: %{customdata[2]:.2f}<extra></extra>",
        )
    )
    yaxis_title = "xG per possession" if column == "xg_per_possession" else "Share of possessions"
    fig.update_layout(
        xaxis_title="Phase of play",
        yaxis_title=yaxis_title,
        showlegend=True,
        margin={"l": 60, "r": 20, "t": 10, "b": 60},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    return fig
