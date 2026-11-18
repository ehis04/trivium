from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from components.data import load, require
from components.layout import page_setup, show_table, stage_name
from components.sidebar import select_context
from trivium.definitions import POINT_DEFINITIONS, POINTS, RESEARCH_QUESTION, STRENGTH
from trivium.charts.tactical import profile_map, profile_outcomes, score_associations, sensitivity_chart


page_setup("Which team styles go with better-than-expected results?")
tournament, selected_team, _ = select_context()
teams = require("tactical_profile_team", "profiles")
summary = require("tactical_profile_summary", "profiles")
axes = require("tactical_profile_axes", "profiles")
sensitivity = require("tactical_profile_sensitivity", "profiles")
if teams is None or summary is None or axes is None or sensitivity is None:
    st.stop()
teams_t = teams.loc[teams["tournament"].eq(tournament)]
axes_t = axes.loc[axes["tournament"].eq(tournament)]

point_definitions = "\n".join(
    f"- **{point}:** {POINT_DEFINITIONS[point].replace('xG', 'xG (expected goals: an estimate of chance quality)').replace('xA', 'xA (expected assists: an estimate of pass quality)')}"
    for point in POINTS
)
st.markdown(
    f"**Question:** {RESEARCH_QUESTION}\n\n"
    f"{point_definitions}\n\n"
    f"{STRENGTH} (Elo is a rating of senior-team strength.)"
)

team_row = teams_t.loc[teams_t["team"].eq(selected_team)]
if not team_row.empty:
    r = team_row.iloc[0]
    cards = st.columns(4)
    cards[0].metric(selected_team, r["profile"].replace(" · ", ", "))
    cards[1].metric("Points above expectation per match", f"{r['overperformance_per_match']:+.2f}")
    cards[2].metric("Reached", stage_name(r.get("actual_stage_name", "")))
    progress = r.get("progress_over_expectation")
    cards[3].metric("Stages beyond expectation", f"{progress:+.2f}" if pd.notna(progress) else "n/a")

st.subheader("How teams compare")
st.plotly_chart(profile_map(teams, tournament, selected_team), width="stretch")
st.caption(
    "Each dot is a team: the horizontal axis is its dominance score and the vertical axis its efficiency score, "
    "both measured in standard deviations (units of typical team-to-team spread) from the tournament average. "
    "The upper-right is above average on both; green means more points than senior-team strength predicted, orange "
    "means fewer. Filled dots mean more resourceful and open dots mean more costly."
)

st.subheader("Results by team style")
profile = team_row.iloc[0]["profile"] if not team_row.empty else None
st.plotly_chart(profile_outcomes(summary, tournament, profile), width="stretch")
st.caption(
    "Each dot is a style's mean points above expectation per match; its horizontal line is the 95% uncertainty "
    "range, based on resampling teams. Blue highlights this team's style, green means above expectation and orange "
    "means below. The horizontal axis is points above expectation and the vertical line at zero means expectation "
    "was met; ranges crossing zero do not show a clear difference."
)

st.subheader("Which scores track better-than-expected results?")
st.plotly_chart(score_associations(axes), width="stretch")
st.caption(
    "Each dot is a profile score (rows), with its link to points above expectation on the horizontal axis; the "
    "horizontal line shows its 95% uncertainty range. Blue is the men's tournament and pink the women's; filled "
    "dots pass both repeatability and later-match checks, while open dots do not. Farther right is a stronger "
    "positive link, but this is not proof that the score caused the result."
)
gate = load("finding_gate")
if gate is not None and "finding_type" in gate:
    rows = gate.loc[gate["finding_type"].eq("tactical profile") & gate["tournament"].eq(tournament)]
    if not rows.empty:
        st.markdown("\n".join(
            f"- {'**Passes the slide gate.** ' if bool(r.passes_gate) else ''}{r.wording}"
            for r in rows.itertuples(index=False)))

with st.expander("Teams by profile"):
    columns = [("team", "Team"), ("profile", "Profile"), ("dominant", "Dominant"), ("efficient", "Efficient"),
               ("resourceful", "Resourceful"), ("overperformance_per_match", "Points above expectation per match"),
               ("actual_stage_name", "Reached"), ("elo", "Strength rating (Elo)")]
    table = teams_t[[c for c, _ in columns if c in teams_t]].rename(columns=dict(columns))
    if "Reached" in table:
        table["Reached"] = table["Reached"].map(stage_name)
    show_table(table.round(2), hide_index=True, width="stretch")

with st.expander("Do the patterns hold under other definitions?"):
    sensitivity_t = sensitivity.loc[sensitivity["tournament"].eq(tournament)]
    st.plotly_chart(sensitivity_chart(sensitivity, tournament), width="stretch")
    st.caption(
        "Each dot is the link between one profile score and points above expectation under a different definition; "
        "the horizontal line is its 95% uncertainty range. Blue, green and pink identify the dominant, efficient and "
        "resourceful scores; a diamond is the main definition and circles are alternatives. The horizontal axis is the "
        "strength and direction of the link, and the dashed line marks no link; farther right is a stronger positive link."
    )
    keeps_sign = sensitivity_t.groupby("score")["same_sign_as_main"].all()
    score_labels = {"dominant": "dominant", "efficient": "efficient", "resourceful": "resourceful"}
    stable_scores = [score_labels.get(score, score) for score in keeps_sign.index[keeps_sign]]
    if stable_scores:
        score_text = ", ".join(stable_scores[:-1]) + (
            f" and {stable_scores[-1]}" if len(stable_scores) > 1 else stable_scores[0]
        )
        st.markdown(f"The {score_text} scores keep their sign under every definition variant.")
    else:
        st.markdown("No score keeps its sign under every definition variant.")
    if not sensitivity_t.empty:
        stability = sensitivity_t.iloc[0]
        st.markdown(
            f"Profiles match the main assignment for {stability['median_threshold_share']:.0%} of teams "
            f"at the median threshold and {stability['leave_one_out_share']:.0%} across leave-one-team-out fits."
        )
