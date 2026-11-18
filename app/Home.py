from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from components import charts
from components.data import load, require
from components.glossary import glossary
from components.layout import expected_stage_label, page_setup, show_table, stage_name
from components.sidebar import select_context
from three_trees.charts.expectation import (
    expected_vs_actual_points,
    expected_vs_actual_stage,
    overperformance_chart,
    overperformance_dumbbell,
)
from three_trees.charts.profiles import profile_comparison_chart
from three_trees.charts.simple import evidence_bars, profile_bars, ranked_bars, scatter_fit
from three_trees.charts.tactical import profile_outcomes
from three_trees.models.answer import point_evidence
from three_trees.definitions import RESEARCH_QUESTION


page_setup("Tournament pathways")

st.markdown(f"**Research question**: {RESEARCH_QUESTION}")
st.write(
    "The dashboard answers it by measuring each team against its senior World Football Elo expectation, "
    "grouping match performance into dominant, resourceful and efficient tactical profiles, and testing "
    "which traits carry teams further in the tournament."
)

st.subheader("Start here")
pages_dir = Path(__file__).resolve().parent / "pages"
page_links_config = [
    (
        "pages/0_Research_question.py",
        "The research question",
        "The core question, what team strength predicts, and how dominance, resourcefulness and efficiency answer it.",
    ),
    (
        "pages/1_Coach_view.py",
        "Coach view",
        "Pick a team: its tactical profile in plain football language and three things to do against it.",
    ),
    (
        "pages/1_Tactical_profiles.py",
        "Tactical profiles",
        "Which tactical profiles beat their strength, the full profile map, and out-of-sample evidence.",
    ),
    (
        "pages/2_Team_report.py",
        "Team report",
        "Single-team tournament summary, tactical profile scores, match-by-match trends, and pitch control.",
    ),
    (
        "pages/3_Match_review.py",
        "Match review",
        "Match review with dominance bars, xG timeline, and pitch zone maps (the Ireland v Panama demo).",
    ),
    (
        "pages/4_Style_explorer.py",
        "Style explorer",
        "Playing styles across teams, style clusters, and tactical trait distributions.",
    ),
    (
        "pages/5_Running_and_load.py",
        "Running and load",
        "Physical metrics, high-speed running, and player workload across positions (men's tournament).",
    ),
    (
        "pages/6_Ask.py",
        "Ask",
        "Ask questions about tournament results, tactical profiles, and team measures in plain English.",
    ),
    (
        "pages/7_Method_and_limits.py",
        "Method and limits",
        "Measure definitions, data coverage across sources, build verification checks, and limits.",
    ),
]

for rel_path, label, description in page_links_config:
    file_name = Path(rel_path).name
    if (pages_dir / file_name).is_file():
        st.page_link(rel_path, label=label, help=description)
        st.caption(description)

tournament, selected_team, _ = select_context()

st.subheader("The answer")
gate = load("finding_gate")
if gate is not None and "finding_type" in gate and "tournament" in gate and "passes_gate" in gate:
    profile_findings = gate.loc[
        gate["finding_type"].eq("tactical profile")
        & gate["tournament"].eq(tournament)
        & gate["passes_gate"].eq(True)
    ]
else:
    profile_findings = pd.DataFrame()

if not profile_findings.empty:
    for _, row in profile_findings.iterrows():
        st.markdown(f"- {row['wording']}")
else:
    st.write("No tactical profile finding passes the evidence gate in this tournament.")

st.markdown(
    "Tactical profiles classify teams on three scores: **dominant** (territory and chances share), "
    "**efficient** (converting dominance into chances while limiting opponent chances), and "
    "**resourceful** (dominance bought at low defensive and running cost)."
)

tactical_teams = load("tactical_profile_team")
highlight_profile = None
if tactical_teams is not None and "tournament" in tactical_teams and "team" in tactical_teams:
    t_match = tactical_teams.loc[
        tactical_teams["tournament"].eq(tournament) & tactical_teams["team"].eq(selected_team)
    ]
    if not t_match.empty and "profile" in t_match:
        highlight_profile = str(t_match.iloc[0]["profile"])

tactical_summary = load("tactical_profile_summary")
if tactical_summary is not None:
    st.plotly_chart(profile_bars(tactical_summary, tournament, highlight_profile), width="stretch")
    st.caption(
        "Simple view. One bar per tactical profile: how many points per match its teams won above what their "
        "strength predicted, on average. Green bars beat their strength, orange bars fell short; the number on each "
        "bar is how many teams had that profile, and the outlined bar is the selected team's profile."
    )
    st.plotly_chart(
        profile_outcomes(tactical_summary, tournament, highlight_profile=highlight_profile),
        width="stretch",
    )
    st.caption(
        "Each dot shows a tactical profile's mean points above senior Elo expectation per match, "
        "with horizontal lines showing the 95% bootstrap confidence interval. Green dots beat expectation (good), "
        "orange dots fell short (bad), and blue highlights the selected team's profile. "
        "Look for profiles to the right of the dashed zero line whose intervals stay clear of zero."
    )

st.page_link("pages/1_Tactical_profiles.py", label="Explore tactical profiles in depth")

summary, matches = require("team_summary", "expectation"), require("matches", "StatsBomb ingest")
if summary is None or matches is None:
    st.stop()
team_summary = summary.loc[summary["tournament"].eq(tournament)].copy()
matches_t = matches.loc[matches["tournament"].eq(tournament)]

luck_summary = load("luck_summary")
if luck_summary is not None:
    luck = luck_summary.loc[luck_summary["tournament"].eq(tournament)]
else:
    luck = None

above = int(team_summary["beyond_luck"].eq("above").sum()) if "beyond_luck" in team_summary else 0
below = int(team_summary["beyond_luck"].eq("below").sum()) if "beyond_luck" in team_summary else 0
chance = float(luck.iloc[0]["expected_by_chance"]) if luck is not None and not luck.empty else 0.0
metrics = st.columns(4)
metrics[0].metric("Teams", f"{team_summary['team'].nunique():,}")
metrics[1].metric("Matches", f"{matches_t['match_id'].nunique():,}")
metrics[2].metric("Above / below luck band", f"{above} / {below}")
metrics[3].metric("Expected beyond luck by chance", f"{chance:.1f}")

bands = load("overperformance_luck")
if bands is None:
    bands = load("team_summary")
st.subheader("Actual against expected points")
every_team = st.toggle("Show every team", key="dumbbell_every_team")
n_teams = len(team_summary) if every_team else 5
st.plotly_chart(
    overperformance_dumbbell(team_summary, tournament, highlight=selected_team, top=n_teams, bottom=n_teams),
    width="stretch",
)
st.caption(
    "Each line connects a team's expected tournament points (grey dot, based on its senior Elo rating) "
    "to its actual points won (coloured dot; blue for the selected team). A coloured dot to the right of the "
    "grey dot means the team beat its strength expectation (good); to the left means it underperformed (bad). "
    "The longer the line, the further the team ended up from its pre-tournament expectation."
)
st.subheader("Every team ranked against its strength")
st.plotly_chart(
    ranked_bars(team_summary, "overperformance_per_match", "Points above Elo expectation per match",
                highlight=selected_team),
    width="stretch",
)
st.caption(
    "Simple view. One bar per team: the points it won per match above (right of the line) or below (left) what its "
    "senior Elo strength predicted. The selected team is blue. Teams at the top went furthest beyond their strength."
)

tournament_teams = load("team_tournament")
if tournament_teams is not None:
    pattern = tournament_teams.loc[tournament_teams["tournament"].eq(tournament)].merge(
        team_summary[["team", "overperformance_per_match"]], on="team", how="inner")
    st.subheader("The pattern in one picture")
    left, right = st.columns(2)
    for column, (measure, label, short) in zip((left, right), (
        ("chances_share", "Share of the match's chances (xG share)", "share of the chances"),
        ("defensive_work_per_dominance", "Defensive actions per point of dominance (the cost of control)",
         "cost of control"),
    )):
        if measure not in pattern:
            continue
        figure, fit = scatter_fit(pattern, measure, "overperformance_per_match", label,
                                  "Points above expectation per match", highlight=selected_team)
        with column:
            st.plotly_chart(figure, width="stretch")
            more = "more" if fit["slope"] > 0 else "fewer"
            st.caption(
                f"Each dot is a team (selected team in blue); the straight line is the line of best fit through "
                f"them. It {'rises' if fit['slope'] > 0 else 'falls'} from left to right: teams with a higher "
                f"{short} won {more} points above what their strength predicted (correlation {fit['r']:+.2f}, "
                f"{int(fit['n'])} teams; 1 would be a perfect straight line, 0 no pattern)."
            )

if bands is not None:
    st.subheader("Points above expectation, with luck bands")
    st.plotly_chart(overperformance_chart(team_summary, tournament, bands), width="stretch")
    st.caption(
        "Each dot shows a team's points above expectation, surrounded by a grey band showing the range of points "
        "expected from match luck alone (the 5th to 95th percentile across simulated tournaments). Dots beyond the grey band "
        "to the right achieved genuine over-performance that luck cannot explain (good); dots beyond to the left genuinely "
        "underperformed (bad). Dots inside the band performed within normal tournament variation."
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Expected and actual points")
        st.plotly_chart(expected_vs_actual_points(team_summary, tournament), width="stretch")
        st.caption(
            "Each dot is a team comparing expected points from senior Elo (horizontal axis) to actual points won (vertical axis). "
            "Dots above the diagonal line won more points than their strength predicted (good); dots below won fewer points than expected (bad). "
            "Teams on the line matched their exact pre-tournament expectation."
        )
    with right:
        progress = require("expected_progress", "expectation")
        if progress is not None:
            st.subheader("Expected and actual stage")
            st.plotly_chart(expected_vs_actual_stage(progress, tournament), width="stretch")
            st.caption(
                "Each dot is a team comparing the expected tournament stage from its Elo rating (horizontal axis) to the stage it actually "
                "reached (vertical axis). Dots above the diagonal dashed line advanced further than their strength predicted (good); "
                "dots below were eliminated earlier than expected (bad)."
            )

st.subheader("Team summary")
show_columns = [
    ("team", "Team"),
    ("elo", "Elo"),
    ("actual_points", "Points"),
    ("expected_points", "Expected points"),
    ("overperformance", "Points above expectation"),
    ("band_low", "Luck band low"),
    ("band_high", "Luck band high"),
    ("actual_stage_name", "Stage reached"),
    ("expected_stage", "Expected stage"),
    ("path_easier_by", "Path easier by"),
]
available_columns = [(column, label) for column, label in show_columns if column in team_summary]
display = team_summary[[column for column, _ in available_columns]].rename(
    columns={column: label for column, label in available_columns}
)
if "Elo" in display:
    display["Elo"] = pd.to_numeric(display["Elo"], errors="coerce").round(0)
if "Stage reached" in display:
    display["Stage reached"] = display["Stage reached"].map(stage_name)
if "Expected stage" in display:
    display["Expected stage"] = display["Expected stage"].map(
        lambda value: expected_stage_label(value, tournament)
    )
for column in ("Points", "Expected points", "Points above expectation", "Luck band low", "Luck band high"):
    if column in display:
        display[column] = pd.to_numeric(display[column], errors="coerce").round(2)
if "Points above expectation" in display:
    display = display.sort_values("Points above expectation", ascending=False)
show_table(display, hide_index=True, width="stretch")

efficiency = load("efficiency_team_tournament")
if efficiency is None:
    require("efficiency_team_tournament", "features")
else:
    efficiency_t = efficiency.loc[efficiency["tournament"].eq(tournament)]
    if "efficiency_index" in efficiency_t and not efficiency_t.empty:
        st.subheader("Efficiency ranking")
        st.plotly_chart(charts.efficiency_rank(efficiency_t, summary=team_summary), width="stretch")
        st.caption(
            "Each horizontal bar shows a team's efficiency index (chances created per unit of dominance cost, standardised across the tournament), "
            "with the most efficient teams at the top. Blue bars indicate teams that won more points than their Elo predicted, while grey bars did not. "
            "Higher positive values mean a team created substantial danger without wasting defensive work or running energy (good)."
        )

profiles = load("profile_comparison")
if profiles is None:
    require("profile_comparison", "profiles")
else:
    profiles_t = profiles.loc[profiles["tournament"].eq(tournament)]
    st.subheader("What separates teams once strength is accounted for")
    st.plotly_chart(evidence_bars(point_evidence(profiles), tournament), width="stretch")
    st.caption(
        "Simple view. One bar per measure: how much higher (right) or lower (left) teams that beat their strength "
        "were than the rest, after accounting for strength, in standard deviations (1 = a large difference). Green "
        "or orange bars are clear differences; grey bars are too small to tell apart from chance."
    )
    st.plotly_chart(profile_comparison_chart(profiles_t, tournament), width="stretch")
    st.caption(
        "Each row compares teams above their Elo expectation to teams below it on a tactical trait, measured in standard deviations (horizontal axis). "
        "Filled dots show the difference after accounting for team strength, with horizontal lines showing the 95% confidence interval; "
        "bold blue markers indicate differences clear of zero (good separator). A dot to the right means teams that beat their strength had higher values on that trait."
    )

glossary()
