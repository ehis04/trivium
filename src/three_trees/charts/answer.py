"""Plotly figures for the evidence behind the three points in the research question."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from three_trees.definitions import POINTS

ACCENT = "#0072B2"
WOMEN = "#CC79A7"
GREY = "#777777"
LIGHT_GREY = "#B8B8B8"
TOURNAMENTS = (("men", "Men", ACCENT), ("women", "Women", WOMEN))


def _figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="simple_white", font={"size": 14})
    return fig


def point_evidence_chart(evidence: pd.DataFrame, point: str | None = None) -> go.Figure:
    """Show strength-adjusted measure differences and 95% intervals for both tournaments."""
    fig = _figure()
    required = {
        "point",
        "part",
        "tournament",
        "measure",
        "label",
        "adjusted_difference_z",
        "adjusted_low",
        "adjusted_high",
        "adjusted_excludes_zero",
    }
    if evidence.empty:
        return fig
    missing = required - set(evidence.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}.")
    if point is not None and point not in POINTS:
        raise ValueError(f"Unknown point: {point!r}.")

    rows = evidence.copy()
    if point is not None:
        rows = rows.loc[rows["point"].eq(point)]
    rows = rows.loc[rows["tournament"].isin(("men", "women"))].copy()
    rows["adjusted_difference_z"] = pd.to_numeric(rows["adjusted_difference_z"], errors="coerce")
    rows = rows.dropna(subset=["adjusted_difference_z"])
    if rows.empty:
        return fig

    point_order = {name: index for index, name in enumerate(POINTS)}
    rows["_point_order"] = rows["point"].map(point_order)
    rows = rows.sort_values(["_point_order", "part", "label", "tournament"], kind="stable")

    cursor = 0
    ticks: list[int] = []
    tick_labels: list[str] = []
    point_y: dict[tuple[str, str], float] = {}
    points = POINTS if point is None else (point,)
    for point_name in points:
        point_rows = rows.loc[rows["point"].eq(point_name)]
        if point_rows.empty:
            continue
        if point is None:
            ticks.append(cursor)
            tick_labels.append(f"<b>{point_name}</b>")
            cursor += 1
        for part in point_rows["part"].drop_duplicates():
            part_rows = point_rows.loc[point_rows["part"].eq(part)]
            ticks.append(cursor)
            tick_labels.append(f"<b>{part}</b>")
            cursor += 1
            for measure_name in part_rows["measure"].drop_duplicates():
                measure_rows = part_rows.loc[part_rows["measure"].eq(measure_name)]
                ticks.append(cursor)
                tick_labels.append(str(measure_rows["label"].iloc[0]))
                for row in measure_rows.itertuples(index=False):
                    point_y[(str(row.tournament), str(measure_name))] = float(cursor)
                cursor += 1
            cursor += 0.35

    for tournament, tournament_label, colour in TOURNAMENTS:
        subset = rows.loc[rows["tournament"].eq(tournament)].copy()
        if subset.empty:
            continue
        offset = -0.14 if tournament == "men" else 0.14
        subset["y"] = [
            point_y[(str(tourney), str(measure_name))] + offset
            for tourney, measure_name in zip(subset["tournament"], subset["measure"])
        ]
        low = pd.to_numeric(subset["adjusted_low"], errors="coerce")
        high = pd.to_numeric(subset["adjusted_high"], errors="coerce")
        estimate = subset["adjusted_difference_z"]
        fig.add_trace(
            go.Scatter(
                x=estimate,
                y=subset["y"],
                mode="markers",
                name=tournament_label,
                marker={
                    "color": colour,
                    "size": 11,
                    "symbol": [
                        "circle" if bool(value) else "circle-open"
                        for value in subset["adjusted_excludes_zero"].fillna(False)
                    ],
                    "line": {"color": colour, "width": 1.5},
                },
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": (high - estimate).clip(lower=0).fillna(0),
                    "arrayminus": (estimate - low).clip(lower=0).fillna(0),
                    "color": colour,
                    "thickness": 1.5,
                },
                customdata=subset[["label", "part", "adjusted_low", "adjusted_high"]],
                hovertemplate=(
                    "%{customdata[0]}<br>%{customdata[1]}<br>"
                    "%{x:.2f} SD<br>95% interval %{customdata[2]:.2f} to "
                    "%{customdata[3]:.2f}<extra>%{fullData.name}</extra>"
                ),
            )
        )

    fig.add_vline(x=0, line_color=GREY, line_dash="dash")
    fig.update_layout(
        margin={"l": 190, "r": 25, "t": 15, "b": 70},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        yaxis={
            "tickmode": "array",
            "tickvals": ticks,
            "ticktext": tick_labels,
            "range": [cursor - 0.5, -0.5],
            "title": "",
        },
        xaxis_title=(
            "Teams above minus below their Elo expectation (standard deviations, 95% interval)"
        ),
    )
    return fig
