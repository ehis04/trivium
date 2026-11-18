from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from three_trees.charts.running import cost_of_dominance, fifa_phase_profile, running_heatmap, shape_profile
from three_trees.tracking.running import (
    EXCLUDE_FILLED_IN_SHARE,
    IMPUTED_SHARE_MAX,
    excluded_matches,
    filled_in_share,
)
from components.data import load, require
from components.layout import page_setup, show_table
from components.sidebar import select_context


page_setup("Running and team workload")
tournament, team, _ = select_context()
if tournament == "women":
    st.info("Men only: FIFA physical and broadcast tracking data cover the men's tournament.")
else:
    team_tournament = require("team_tournament", "features")
    running = require("running_player_match", "tracking")
    team_match_provider = require("team_match_provider", "tracking")

    st.subheader("Running needed to control the match")
    if team_tournament is not None and team is not None:
        tournament_rows = team_tournament.loc[team_tournament["tournament"].eq(tournament)]
        st.plotly_chart(cost_of_dominance(tournament_rows, team), width="stretch")
        st.caption(
            "Each dot is a team: farther right means more match control (dominance: control of contests, territory "
            "and chances), and higher means more "
            "high-speed running without the ball (over 20 km/h) per match. Blue marks the selected team, grey "
            "marks others, and dashed lines show tournament medians; the lower-right is more control at lower "
            "running cost, though less running is not useful if it means losing pressure."
        )

    if running is not None:
        tournament_running = running.loc[running["tournament"].eq(tournament)].copy()
        excluded = excluded_matches(tournament_running)
        bad_match_ids = excluded["match_id"].tolist()
        usable_running = tournament_running.loc[~tournament_running["match_id"].isin(bad_match_ids)]

        if team is not None:
            shares = filled_in_share(tournament_running)
            matches = load("matches")
            if matches is None:
                require("matches", "statsbomb")
            elif not shares.empty:
                selected_matches = matches.loc[
                    matches["tournament"].eq(tournament)
                    & (matches["home_team"].eq(team) | matches["away_team"].eq(team))
                ].copy()
                selected_matches["opponent"] = selected_matches.apply(
                    lambda row: row["away_team"] if row["home_team"] == team else row["home_team"],
                    axis=1,
                )
                warnings = selected_matches.merge(
                    shares,
                    on=["tournament", "match_id"],
                    how="inner",
                    validate="one_to_one",
                )
                warnings = warnings.loc[warnings["filled_in_share"] > IMPUTED_SHARE_MAX]
                if not warnings.empty:
                    messages = []
                    for row in warnings.itertuples(index=False):
                        message = (
                            f"{row.opponent} ({row.competition_stage}): "
                            f"{row.filled_in_share:.0%} of player positions filled in by the provider"
                        )
                        if row.filled_in_share > EXCLUDE_FILLED_IN_SHARE:
                            message += "; left out of the tracking measures."
                        else:
                            message += "; running figures for this match are less reliable."
                        messages.append(message)
                    st.warning("\n\n".join(messages))

        st.subheader("Where high-speed running (over 20 km/h) happens")
        if team is not None:
            st.plotly_chart(running_heatmap(usable_running, team), width="stretch")
            st.caption(
                "Each cell shows the selected team's high-speed running (over 20 km/h) per 90 minutes for a "
                "position group and time with or without the ball; columns are own, middle and attacking thirds, "
                "and rows are possession states. Darker blue means more running. Compare the pattern with the team's "
                "plan and tournament demands: more is not automatically better. "
                "Matches with more than 30% of positions filled in by the provider are excluded."
            )

        position = usable_running.groupby(
            ["tournament", "team", "position_group", "state"], as_index=False
        )[["minutes", "distance_m", "hsr_m", "sprint_m"]].sum()
        usable_minutes = position["minutes"].where(position["minutes"] > 0)
        position["distance_km_per_90"] = position["distance_m"] / usable_minutes * 90 / 1000
        position["hsr_km_per_90"] = position["hsr_m"] / usable_minutes * 90 / 1000
        position["sprint_km_per_90"] = position["sprint_m"] / usable_minutes * 90 / 1000
        shown = position.drop(columns=["distance_m", "hsr_m", "sprint_m", "minutes"])
        shown = shown.rename(columns={
            "team": "Team",
            "position_group": "Position group",
            "state": "When the team had the ball",
            "distance_km_per_90": "Distance (kilometres per 90 minutes)",
            "hsr_km_per_90": "High-speed running (kilometres per 90 minutes)",
            "sprint_km_per_90": "Sprinting (kilometres per 90 minutes)",
        })
        shown["When the team had the ball"] = shown["When the team had the ball"].map({
            "with_ball": "With the ball", "without_ball": "Without the ball", "out_of_play": "Ball out of play",
        })
        with st.expander("Running per 90 minutes by position group and possession state"):
            show_table(shown.round(2), hide_index=True, width="stretch")

    if team_match_provider is not None and team is not None:
        st.subheader("How play was split into phases")
        st.plotly_chart(fifa_phase_profile(team_match_provider, team), width="stretch")
        st.caption(
            "Each row is a phase named by FIFA; blue dots show this team's share of phase time and grey dots show "
            "the tournament average. The horizontal axis is the share of time (farther right means more time in "
            "that phase), which describes style rather than success."
        )
        st.subheader("Team shape when defending")
        st.plotly_chart(shape_profile(team_match_provider, team), width="stretch")
        st.caption(
            "Blue dots show this team's average line height, team length and width; grey dots show tournament "
            "averages, and the light band spans the range across teams. The horizontal axis is metres: farther "
            "right means a larger value, but the best shape depends on the team's plan."
        )

team_match = require("team_match", "features")
if team_match is not None:
    match_rows = team_match.loc[team_match["tournament"].eq(tournament)]
    rotation_columns = [
        column for column in ("minutes_outside_top11_share", "starters_changed") if column in match_rows
    ]
    if rotation_columns:
        rotation = match_rows.groupby("team", as_index=False)[rotation_columns].mean()
        if "minutes_outside_top11_share" in rotation:
            rotation["Share of minutes played by the regular starting 11"] = (
                1 - rotation.pop("minutes_outside_top11_share")
            )
        st.subheader("How the team shares minutes and changes starters")
        rotation = rotation.rename(columns={
            "team": "Team",
            "tournament": "Tournament",
            "starters_changed": "Number of starters changed per match",
        })
        show_table(rotation.round(2), hide_index=True, width="stretch")

if tournament == "men":
    efficiency = load("efficiency_position_group")
    if efficiency is None:
        require("efficiency_position_group", "features")
    else:
        st.subheader("Running cost by position group")
        display = efficiency.loc[efficiency["tournament"].eq(tournament)].rename(columns={
            "tournament": "Tournament", "team": "Team", "position_group": "Position group",
            "matches": "Matches", "hsr_without_ball_m_per_match":
                "High-speed running without the ball (metres per match)",
            "cost_per_dominance": "Running per point of match control",
            "cost_z": "Running-cost score (higher means lower cost)",
        })
        show_table(
            display,
            hide_index=True,
            width="stretch",
        )
