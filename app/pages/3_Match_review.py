from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from components.data import load, require
from components.formations import shape_png
from components.pitch import zone_panels
from components.sidebar import select_context
from components.layout import page_setup, show_table
from three_trees.definitions import POINT_DEFINITIONS
from three_trees.charts.formations import formation_timeline
from three_trees.charts.match import dominance_bars, xg_timeline
from three_trees.tracking.running import EXCLUDE_FILLED_IN_SHARE, IMPUTED_SHARE_MAX, filled_in_share


page_setup("Match review")
tournament, selected_team, match_id = select_context()
if selected_team is None or match_id is None:
    st.stop()
matches = require("matches", "StatsBomb ingest")
if matches is None:
    st.stop()
match_rows = matches.loc[matches["tournament"].eq(tournament) & matches["match_id"].eq(match_id)]
if match_rows.empty:
    st.info("This match is not available in the current build.")
    st.stop()
match = match_rows.iloc[0]
st.header(
    f"{match['home_team']} {int(match['home_score'])}-{int(match['away_score'])} {match['away_team']}"
)
opponent = match["away_team"] if match["home_team"] == selected_team else match["home_team"]
st.caption(f"{match['competition_stage']}, {match['match_date']}. Selected team: {selected_team}.")

team_match = require("team_match", "features")
dominance = None
if team_match is not None:
    dominance = team_match.loc[
        team_match["tournament"].eq(tournament) & team_match["match_id"].eq(match_id)
    ].copy()
    st.subheader("Which team had more match control?")
    st.plotly_chart(dominance_bars(dominance, selected_team), width="stretch")
    dominance_definition = POINT_DEFINITIONS["Dominance"].replace(
        "non-penalty expected goals, xG",
        "non-penalty xG (expected goals: an estimate of chance quality)",
    )
    st.caption(
        f"{dominance_definition} Each bar is the selected team's share of that part; the horizontal axis runs from "
        "0% to 100%, with 50% meaning even. Blue means the selected team led and orange means the opponent led; "
        "a larger share means more control in that part, but does not guarantee a win."
    )

shots = load("shot_timeline")
if shots is None:
    require("shot_timeline", "features")
else:
    st.subheader("Chances created through the match")
    match_shots = shots.loc[shots["tournament"].eq(tournament) & shots["match_id"].eq(match_id)]
    st.plotly_chart(xg_timeline(match_shots, selected_team, opponent), width="stretch")
    st.caption(
        "Each step line adds the team's xG (expected goals: an estimate of chance quality) as shots happen; blue is "
        f"{selected_team} and orange is {opponent}. The horizontal axis is minutes played and the vertical axis is "
        "cumulative xG; goal markers show when goals were scored and the dotted line marks half-time. A higher line "
        "means more or better-valued chances, not necessarily more goals."
    )

zones = load("team_match_zone")
if zones is None:
    require("team_match_zone", "features")
else:
    st.subheader("Territory and contests by pitch zone")
    zone_measure = st.radio(
        "Show on the pitch",
        ["Territory share", "Contests won share"],
        horizontal=True,
        key="match_zone_measure",
    )
    measure = "territory_share" if zone_measure == "Territory share" else "contests_share"
    match_zones = zones.loc[
        zones["tournament"].eq(tournament) & zones["match_id"].eq(match_id) & zones["team"].eq(selected_team)
    ]
    panels = [(f"{selected_team} v {opponent}", match_zones)]
    tournament_zones = load("team_tournament_zone")
    if tournament_zones is not None:
        team_zones = tournament_zones.loc[
            tournament_zones["tournament"].eq(tournament) & tournament_zones["team"].eq(selected_team)
        ]
        panels.append((f"{selected_team}, whole tournament", team_zones))
    zone_panels(panels, measure)
    st.caption(
        "Each tile is one of 18 pitch zones; the first panel is this match and the second, when available, is the "
        "team's tournament pattern. The pitch is shown in the team's attacking direction from left to right. Blue "
        "means the team had the larger share of the selected measure, orange the opponent, and grey is near even; "
        "more territory or contests won in a zone can help explain where control came from."
    )

st.subheader("Team shape during the match")
listed = load("formation_listed")
if listed is None:
    require("formation_listed", "features")
else:
    listed_match = listed.loc[
        listed["tournament"].eq(tournament)
        & listed["match_id"].eq(match_id)
        & listed["team"].eq(selected_team)
    ]
    if listed_match.empty:
        st.caption("Listed formation: not available for this match.")
    else:
        listed_row = listed_match.iloc[0]
        starting_value = listed_row["starting_formation"]
        starting = starting_value if pd.notna(starting_value) else "not recorded"
        changes_value = listed_row["formation_changes"]
        changes = int(changes_value) if pd.notna(changes_value) else 0
        change_text = "no formation changes" if changes == 0 else (
            "changed once" if changes == 1 else f"changed {changes} times"
        )
        st.caption(f"Listed formation: {starting}, {change_text}.")

if tournament == "women":
    st.info("Men only: detected formations use the men's tracking data.")
else:
    positions = load("positions_window")
    if positions is None:
        require("positions_window", "tracking")
    else:
        windows = load("formation_team_window")
        if windows is None:
            require("formation_team_window", "tracking")
        else:
            team_windows = windows.loc[
                windows["tournament"].eq(tournament)
                & windows["match_id"].eq(match_id)
                & windows["team"].eq(selected_team)
            ]
            if team_windows.empty:
                st.info("No tracking shape is available for this team and match.")
            else:
                st.plotly_chart(formation_timeline(windows, selected_team, match_id), width="stretch")
                st.caption(
                    "Each colored bar shows the detected formation for a five-minute span; the two rows compare the "
                    "team with and without the ball, and the horizontal axis is match minute. Colors identify "
                    "formations and the dashed line marks half-time; a change shows a shape adjustment, not a "
                    "better-or-worse score by itself."
                )
                team_positions = positions.loc[
                    positions["tournament"].eq(tournament)
                    & positions["match_id"].eq(match_id)
                    & positions["team"].eq(selected_team)
                ]
                states = {"With the ball": "with_ball", "Without the ball": "without_ball"}
                state_label = st.radio("Average shape", list(states), horizontal=True, key="formation_state")
                available_windows = (
                    team_positions[["period", "window", "window_start_min"]]
                    .drop_duplicates()
                    .sort_values(["period", "window"], kind="stable")
                )
                options = [None, *[
                    (int(row.period), int(row.window)) for row in available_windows.itertuples(index=False)
                ]]
                labels = {
                    (int(row.period), int(row.window)): (
                        f"{float(row.window_start_min):g}-{float(row.window_start_min) + 5:g} min"
                    )
                    for row in available_windows.itertuples(index=False)
                }
                window = st.selectbox(
                    "5-minute window",
                    options,
                    format_func=lambda value: "Whole match" if value is None else labels[value],
                    key="formation_window",
                )
                st.image(
                    shape_png(team_positions, selected_team, int(match_id), states[state_label], window),
                    width="stretch",
                )
                st.caption(
                    "Each dot is an average position within a position group; colors distinguish defenders, "
                    "midfielders and forwards, and grey lines connect the team's approximate lines. Compare the "
                    "team's shape with or without the ball and across the chosen time window; this is a shape view, "
                    "not a score where higher or lower is automatically better."
                )

performance = load("overperformance")
expected = load("expected_results")
if performance is None:
    require("overperformance", "expectation")
else:
    rows = performance.loc[
        performance["tournament"].eq(tournament) & performance["match_id"].eq(match_id)
    ].copy()
    if expected is not None:
        probs = expected.loc[
            expected["tournament"].eq(tournament) & expected["match_id"].eq(match_id),
            ["tournament", "match_id", "team", "p_loss", "p_draw", "p_win", "expected_points"],
        ]
        rows = rows.drop(columns=["p_loss", "p_draw", "p_win", "expected_points"], errors="ignore").merge(
            probs,
            on=["tournament", "match_id", "team"],
            how="left",
            validate="one_to_one",
        )
    with st.expander("Expected result and actual points", expanded=True):
        show = [
            column
            for column in (
                "team", "opponent", "p_win", "p_draw", "p_loss", "expected_points",
                "actual_points", "overperformance",
            )
            if column in rows
        ]
        display = rows[show].rename(columns={
            "team": "Team",
            "opponent": "Opponent",
            "p_win": "Win chance",
            "p_draw": "Draw chance",
            "p_loss": "Loss chance",
            "expected_points": "Expected points",
            "actual_points": "Actual points",
            "overperformance": "Points above expectation",
        })
        show_table(display.round(3), hide_index=True, width="stretch")

if dominance is not None:
    with st.expander("Match control and defensive effort"):
        columns = [
            column
            for column in (
                "team", "contests_share", "territory_share", "chances_share", "dominance",
                "defensive_work_per_dominance", "hsr_without_ball_per_dominance",
            )
            if column in dominance
        ]
        show_table(dominance[columns].rename(columns={
            "team": "Team",
            "contests_share": "Share of contests won",
            "territory_share": "Share of territory in final third (field tilt)",
            "chances_share": "Share of chance quality (xG)",
            "dominance": "Overall match control",
            "defensive_work_per_dominance": "Defensive work per point of control",
            "hsr_without_ball_per_dominance": "High-speed running (above 20 km/h) per point of control",
        }).round(3), hide_index=True, width="stretch")

running = load("running_player_match")
if tournament == "women":
    st.info("Men only: tracking data covers the men's tournament.")
elif running is not None:
    selected = running.loc[
        running["tournament"].eq(tournament) & running["match_id"].eq(match_id)
    ]
    if selected.empty:
        st.info("No running data is available for this match.")
    else:
        filled = float(filled_in_share(selected)["filled_in_share"].iloc[0])
        if filled > EXCLUDE_FILLED_IN_SHARE:
            st.warning(f"{filled:.0%} of player positions in this match were filled in by the tracking provider; "
                       "its tracking running and shape figures are left out of the models.")
        elif filled > IMPUTED_SHARE_MAX:
            st.warning(f"{filled:.0%} of player positions in this match were filled in by the tracking provider; "
                       "its running figures are less reliable.")
        grouped = selected.groupby(["team", "position_group", "state"], as_index=False)[
            ["minutes", "distance_m", "hsr_m", "sprint_m"]
        ].sum()
        grouped["distance_m"] = grouped["distance_m"] / 1000
        grouped["hsr_m"] = grouped["hsr_m"] / 1000
        grouped["sprint_m"] = grouped["sprint_m"] / 1000
        with st.expander("Running by position group (men)"):
            grouped["state"] = grouped["state"].map(
                {"with_ball": "With the ball", "without_ball": "Without the ball", "out_of_play": "Out of play"}
            ).fillna(grouped["state"])
            show_table(grouped.rename(columns={
                "team": "Team", "position_group": "Position group", "state": "Ball state",
                "minutes": "Minutes", "distance_m": "Distance (km)", "hsr_m": "High-speed running (km)",
                "sprint_m": "Sprinting (km)",
            }).round(2), hide_index=True, width="stretch")
else:
    require("running_player_match", "tracking")
