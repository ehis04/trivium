from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from components.data import load, require
from components.layout import page_setup, show_table
from components.sidebar import select_context
from three_trees.charts.formations import formation_results
from three_trees.measures import available, measures


page_setup("How teams play")
tournament, selected_team, _ = select_context()
style_types = load("style_types")
style_profiles = load("style_profiles")
style_overperformance = load("style_overperformance")
if style_types is None:
    require("style_types", "styles")
else:
    st.subheader("Different ways teams play")
    profiles = require("style_profiles", "styles")
    if profiles is not None and {"style_id", "style_label"}.issubset(profiles.columns):
        style = style_types.loc[style_types["tournament"].eq(tournament)]
        profile = profiles.loc[profiles["tournament"].eq(tournament)]
        measures_z = [column for column in profile if column.endswith("_z")]
        measure_labels = measures(tournament=tournament).set_index("column")["label"].to_dict()
        if len(measures_z) >= 2 and not style.empty:
            axis_left, axis_right = st.columns(2)
            with axis_left:
                measure_x = st.selectbox("Horizontal measure", measures_z, key="style_measure_x")
            with axis_right:
                measure_y = st.selectbox(
                    "Vertical measure",
                    measures_z,
                    index=1 if len(measures_z) > 1 else 0,
                    key="style_measure_y",
                )
            chart_data = style.merge(
                profile[["style_id", "style_label", *measures_z]],
                on=["style_id", "style_label"],
            )
            figure = px.scatter(
                chart_data,
                x=measure_x,
                y=measure_y,
                color="style_label",
                hover_name="team",
                labels={
                measure_x: f"{measure_labels.get(measure_x.removesuffix('_z'), measure_x.removesuffix('_z').replace('_', ' ').title())} (difference from tournament average)",
                measure_y: f"{measure_labels.get(measure_y.removesuffix('_z'), measure_y.removesuffix('_z').replace('_', ' ').title())} (difference from tournament average)",
                },
                template="simple_white",
            )
            figure.update_layout(font={"size": 14})
            st.plotly_chart(figure, width="stretch")
            st.caption(
                "Each dot is a team; colours show groups of teams with similar styles. The axes compare the selected "
                "measures with the tournament average: right and higher mean more of that measure, but neither is "
                "automatically better for every game plan."
            )
        profile_display = profile.rename(columns={
            "tournament": "Tournament", "style_id": "Style group", "style_label": "Style description",
            "n_teams": "Teams",
            **{
                column: f"{measure_labels.get(column.removesuffix('_z').removesuffix('_mean'), column)}"
                    f" ({'difference from tournament average' if column.endswith('_z') else 'team average'})"
                for column in profile.columns if column.endswith(("_z", "_mean"))
            },
        })
        show_table(profile_display, hide_index=True, width="stretch")
        overperformance = load("style_overperformance")
        if overperformance is None:
            require("style_overperformance", "models")
        else:
            overperformance_display = overperformance.loc[
                overperformance["tournament"].eq(tournament)
            ].rename(columns={
                "tournament": "Tournament", "style_id": "Style group", "style_label": "Style description",
                "n_teams": "Teams", "overperformance_per_match": "Points above strength's expectation per match",
                "low": "Lower likely value", "high": "Upper likely value",
                "excludes_zero": "Clear difference from expectation",
            })
            show_table(
                overperformance_display,
                hide_index=True,
                width="stretch",
            )

st.subheader("Team shapes and results")
formation_summary = load("team_summary")
if formation_summary is None:
    require("team_summary", "expectation")
detected = load("formation_team_match")
if detected is None:
    require("formation_team_match", "tracking")
listed = load("formation_listed")
if listed is None:
    require("formation_listed", "features")

formation_rows = []
if formation_summary is not None:
    if detected is not None:
        without_ball = detected.loc[detected["state"].eq("without_ball")]
        if without_ball.empty:
            st.info("No team shapes could be determined from men's tracking.")
        else:
            st.markdown("Most common shape when defending (from tracking, men)")
            st.plotly_chart(
                formation_results(without_ball, formation_summary, "main_formation"),
                width="stretch",
            )
            st.caption(
                "The vertical axis lists formations (starting arrangements); each blue dot is the average points a "
                "team earned above its pre-tournament team-strength estimate "
                "(Elo) for that defending formation; the light line shows the likely range "
                "and the vertical line marks no difference. Dots farther right are better results than expected, "
                "but overlapping ranges are not a clear gap."
            )
            detected_rows = without_ball.sort_values("match_id", kind="stable")
            for (competition, team), group in detected_rows.groupby(["tournament", "team"], sort=False):
                counts = group["main_formation"].value_counts()
                first_seen = group["main_formation"].drop_duplicates().tolist()
                formation_rows.append({
                    "Source": "Detected without the ball",
                    "Tournament": competition,
                    "Team": team,
                    "Most common team shape": next(shape for shape in first_seen if counts[shape] == counts.max()),
                })
    if listed is not None:
        st.markdown("Starting formation listed in match data (men and women)")
        st.plotly_chart(
            formation_results(listed, formation_summary, "main_formation"),
            width="stretch",
        )
        st.caption(
            "The vertical axis lists formations (starting arrangements); each blue dot is the average points teams "
            "earned above their pre-tournament team-strength estimate "
            "(Elo) for that listed formation; the light line shows the likely range and the "
            "vertical line marks no difference. Dots farther right are better results than expected, but "
            "overlapping ranges are not a clear gap."
        )
        listed_rows = listed.sort_values("match_id", kind="stable")
        for (competition, team), group in listed_rows.groupby(["tournament", "team"], sort=False):
            group = group.dropna(subset=["main_formation"])
            if group.empty:
                continue
            counts = group["main_formation"].value_counts()
            first_seen = group["main_formation"].drop_duplicates().tolist()
            formation_rows.append({
                "Source": "StatsBomb listed",
                "Tournament": competition,
                "Team": team,
                "Most common team shape": next(shape for shape in first_seen if counts[shape] == counts.max()),
            })
    if formation_rows:
        formations = pd.DataFrame(formation_rows).drop_duplicates(
            ["Source", "Tournament", "Team"], keep="first"
        )
        scores = formation_summary[["tournament", "team", "overperformance_per_match"]].rename(
            columns={"tournament": "Tournament", "team": "Team",
                     "overperformance_per_match": "Points above expectation per match"}
        )
        formations = formations.merge(scores, on=["Tournament", "Team"], how="left", validate="many_to_one")
        show_table(
            formations.sort_values(["Source", "Tournament", "Team"], ignore_index=True).round(3),
            hide_index=True,
            width="stretch",
        )

st.subheader("Compare two teams' measures")
team_match = require("team_match", "features")
elo = load("elo_pre_tournament")
if team_match is None or elo is None:
    st.stop()
teams = sorted(elo.loc[elo["tournament"].eq(tournament), "team"].astype(str).unique())
if not teams:
    st.info("There are no teams available for this tournament.")
    st.stop()
left, right = st.columns(2)
with left:
    default_a = teams.index(selected_team) if selected_team in teams else 0
    team_a = st.selectbox("First team", teams, index=default_a)
with right:
    default_b = 1 if len(teams) > 1 else 0
    team_b = st.selectbox("Second team", teams, index=default_b)
table = team_match.loc[team_match["tournament"].eq(tournament)]
registry = available(table, measures(tournament=tournament, table="team_match"))
if registry.empty:
    st.info("No comparable team measures have been built.")
    st.stop()
records = []
for measure in registry.itertuples(index=False):
    values = pd.to_numeric(table.groupby("team")[measure.column].mean(), errors="coerce")
    mean, sd = float(values.mean()), float(values.std(ddof=0))
    for team in (team_a, team_b):
        value = float(values.loc[team]) if team in values.index and pd.notna(values.loc[team]) else np.nan
        rank_values = values.rank(ascending=not bool(measure.higher_is_better), method="min")
        rank = (
            int(rank_values.loc[team])
            if measure.higher_is_better is not None
            else None
        )
        records.append(
            {
                "Measure": measure.label,
                "Team": team,
                "Value": value,
                "Rank": rank,
                "Standardised value": (value - mean) / sd if sd and pd.notna(value) else np.nan,
                "Unit": measure.unit,
                "Higher is better": measure.higher_is_better,
            }
        )
comparison = pd.DataFrame(records)
value_table = comparison.pivot(index=["Measure", "Unit"], columns="Team", values="Value")
rank_table = comparison.pivot(index=["Measure", "Unit"], columns="Team", values="Rank")
value_table.columns = [f"{team} value" for team in value_table.columns]
rank_table.columns = [f"{team} rank" for team in rank_table.columns]
comparison_table = value_table.join(rank_table).reset_index()
show_table(
    comparison_table,
    hide_index=True,
    width="stretch",
)
figure = px.scatter(
    comparison,
    x="Standardised value",
    y="Measure",
    color="Team",
    symbol="Team",
    color_discrete_sequence=["#0072B2", "#777777"],
    template="simple_white",
    labels={"Standardised value": "Difference from tournament average (standard deviations)"},
)
figure.update_layout(font={"size": 14})
st.plotly_chart(figure, width="stretch")
st.caption(
    "Each dot compares one team's average measure with the tournament average; colour identifies the team and the "
    "vertical axis lists measures. Farther right means more of a measure, which is better only when that measure "
    "is desirable; the table above gives the actual values and each team's place among the teams."
)
