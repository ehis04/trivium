from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from components.data import require
from components.layout import page_setup
from components.sidebar import select_context
from trivium.charts.tactical import team_scores
from trivium.coach import scouting_report
from trivium.config import load_config
from trivium.definitions import POINT_DEFINITIONS, POINTS, RESEARCH_QUESTION, STRENGTH


page_setup("Coach view: how to play against them")
tournament, team, _ = select_context()
if team is None:
    st.stop()

team_tournament = require("team_tournament", "features")
if team_tournament is None:
    st.stop()
profile_teams = require("tactical_profile_team", "profiles")
profile_summary = require("tactical_profile_summary", "profiles")

st.caption(RESEARCH_QUESTION)
report = scouting_report(load_config(), team, tournament)
st.subheader("Scouting card")
st.markdown(f"**Profile:** {report['profile']}")
st.write(report["summary"])

st.subheader("Three things to do against them")
if report["game_plan"]:
    for number, instruction in enumerate(report["game_plan"], start=1):
        st.markdown(f"### {number}. {instruction}")
else:
    st.info("There are not enough standout team measures to prepare a game plan.")

st.subheader("Team profile")
if profile_teams is not None:
    st.plotly_chart(team_scores(profile_teams, tournament, team), width="stretch")
    st.caption(
        "Bar length shows how far the team sits from the tournament average; green is above average, orange is below, "
        "and zero is average. The chart uses standard deviations, a way to measure distance from the average. "
        "Higher scores mean stronger dominance, resourcefulness or efficiency."
    )
else:
    st.info("The three-score tactical profile is not available in this build.")

area_order = ("Control", "Spending and cost", "Chances", "Shape", "Phases of play")
for area in area_order:
    area_notes = [note for note in report["notes"] if note["area"] == area]
    if not area_notes:
        continue
    st.subheader(area)
    for note in area_notes:
        st.markdown(f"**What to expect:** {note['observation']}")
        st.markdown(f"**What to do:** {note['recommendation']}")
        st.caption(note["evidence"])

with st.expander("What the three points mean"):
    st.markdown(f"**Strength baseline:** {STRENGTH} Elo is an international team rating.")
    for point in POINTS:
        st.markdown(f"**{point}:** {POINT_DEFINITIONS[point]}")


def _markdown_card() -> str:
    lines = [
        f"# Scouting card: {report['team']}",
        "",
        f"**Tournament:** {report['tournament']}",
        f"**Profile:** {report['profile']}",
        "",
        report["summary"],
        "",
        "## Three things to do against them",
    ]
    lines.extend(f"{number}. {instruction}" for number, instruction in enumerate(report["game_plan"], start=1))
    for area in area_order:
        area_notes = [note for note in report["notes"] if note["area"] == area]
        if not area_notes:
            continue
        lines.extend(["", f"## {area}"])
        for note in area_notes:
            lines.extend([
                "",
                f"**What to expect:** {note['observation']}",
                f"**What to do:** {note['recommendation']}",
                f"**Evidence:** {note['evidence']}",
            ])
    return "\n".join(lines) + "\n"


st.download_button(
    "Download scouting card as Markdown",
    _markdown_card(),
    file_name=f"{tournament}_{team.replace(' ', '_')}_scouting_card.md",
    mime="text/markdown",
)
