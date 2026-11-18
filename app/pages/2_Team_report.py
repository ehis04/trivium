from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from components.charts import comparison_chart, match_trend
from components.data import load, require
from components.findings import build_findings
from components.layout import expected_stage_label, format_points, page_setup, show_table, stage_name
from components.pitch import zone_map
from components.sidebar import select_context
from three_trees.charts.tactical import team_scores
from three_trees.charts.team import phase_dot_plot, tournament_lines
from three_trees.measures import available, measures


page_setup("Team report")
tournament, team, _ = select_context()
if team is None:
    st.stop()
summary = require("team_summary", "expectation")
team_match = require("team_match", "features")
elo = load("elo_pre_tournament")
if summary is None or team_match is None:
    st.stop()
team_summary = summary.loc[summary["tournament"].eq(tournament) & summary["team"].eq(team)]
if team_summary.empty:
    st.info("No team summary is available for this selection.")
    st.stop()
headline = team_summary.iloc[0]
stage = stage_name(headline.get("actual_stage_name", "Not built yet"))
expected_stage = headline.get("expected_stage", None)
overperformance = headline.get("overperformance", None)
band_low, band_high = headline.get("band_low"), headline.get("band_high")
beyond_luck = headline.get("beyond_luck", "")
rank = "Not available"
if elo is not None:
    ranked = elo.loc[elo["tournament"].eq(tournament) & elo["team"].eq(team)]
    if not ranked.empty:
        rank = f"{int(ranked.iloc[0]['elo_rank_in_tournament'])} of {elo['tournament'].eq(tournament).sum()}"
verdict = "beyond luck" if beyond_luck in ("above", "below") else "within what luck allows"
st.header(f"{team} · {stage}")
headline_metrics = st.columns(4)
headline_metrics[0].metric("Strength rank (Elo)", rank)
headline_metrics[1].metric("Expected stage", expected_stage_label(expected_stage, tournament))
headline_metrics[2].metric("Points above expectation", format_points(overperformance))
headline_metrics[3].metric("Range allowed by chance", f"{format_points(band_low)} to {format_points(band_high)}")
st.caption(
    f"{team} finished {stage.lower()}, {verdict}. Points above expectation compare actual points with those "
    "expected from senior Elo (a rating of national-team strength); the range shows how much chance alone could vary."
)
st.caption(
    "xG (expected goals) estimates chance quality; xA (expected assists) estimates the chance quality created by a "
    "pass. OBV (on-ball value) estimates how actions change a team's chance of scoring or conceding."
)

st.subheader("Team style profile")
profile_teams = load("tactical_profile_team")
profile_summary = load("tactical_profile_summary")
if profile_teams is not None and profile_summary is not None:
    team_prof = profile_teams.loc[
        profile_teams["tournament"].eq(tournament) & profile_teams["team"].eq(team)
    ]
    if not team_prof.empty:
        p_row = team_prof.iloc[0]
        prof_name = str(p_row.get("profile", ""))
        st.markdown(f"**Profile:** {prof_name}")
        st.plotly_chart(team_scores(profile_teams, tournament, team), width="stretch")
        st.caption(
            "Each bar is one of the team's three scores; the horizontal axis shows its distance from the tournament "
            "average in standard deviations (units of spread). Green means above average, orange below; the zero line "
            "is average, so farther right means a stronger score."
        )
        summary_prof = profile_summary.loc[
            profile_summary["tournament"].eq(tournament) & profile_summary["profile"].eq(prof_name)
        ]
        if not summary_prof.empty:
            s_row = summary_prof.iloc[0]
            mean_over = float(s_row["overperformance_per_match"])
            low = float(s_row["low"])
            high = float(s_row["high"])
            n = int(s_row["n_teams"])
            st.write(
                f"Teams with the {prof_name} profile averaged {mean_over:+.2f} points above expectation per match "
                f"(95% interval {low:+.2f} to {high:+.2f}, n={n})."
            )
        else:
            st.write(f"Teams with the {prof_name} profile had no summary available.")
    else:
        st.info("No tactical profile recorded for this team.")
else:
    st.info("Tactical profile data is not built in this build.")

tournament_rows = team_match.loc[team_match["tournament"].eq(tournament)]
team_rows = tournament_rows.loc[tournament_rows["team"].eq(team)].copy()
registry = available(team_rows, measures(tournament=tournament, table="team_match"))
measure_columns = registry["column"].drop_duplicates().tolist()
all_team_means = tournament_rows.groupby("team")[measure_columns].mean()


@st.cache_data
def stage_boundaries(tournament_name: str, match_rows: pd.DataFrame, match_performance: pd.DataFrame) -> dict[int, str]:
    """Return the obvious stage change markers for each match number in the tournament."""
    if match_performance is None or match_rows.empty or match_performance.empty:
        return {}
    stage_lookup = (
        match_rows.loc[match_rows["tournament"].eq(tournament_name), ["tournament", "match_id", "match_number"]]
        .drop_duplicates()
        .merge(
            match_performance.loc[match_performance["tournament"].eq(tournament_name), ["tournament", "match_id", "competition_stage"]].drop_duplicates(),
            on=["tournament", "match_id"],
            how="left",
        )
    )
    stage_lookup = stage_lookup.dropna(subset=["competition_stage"]).copy()
    stage_lookup["match_number"] = pd.to_numeric(stage_lookup["match_number"], errors="coerce")
    stage_lookup = stage_lookup.dropna(subset=["match_number"])
    if stage_lookup.empty:
        return {}
    counts = (
        stage_lookup.groupby(["match_number", "competition_stage"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    modal = counts.sort_values(["match_number", "count"], ascending=[True, False], kind="stable").drop_duplicates("match_number")
    stages = {int(row["match_number"]): str(row["competition_stage"]).replace(" Stage", "")
              for _, row in modal.iterrows()}
    changes, previous = {}, None
    for number in sorted(stages):
        if stages[number] != previous:
            changes[number] = stages[number]
            previous = stages[number]
    return changes


@st.cache_data
def tournament_measure_options(tournament_name: str, match_rows: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """All available line-chart measures and their registry columns for the selected tournament."""
    options = ["Points above expectation"]
    mapping: dict[str, str] = {"Points above expectation": "overperformance"}
    available_measures = available(match_rows, measures(tournament=tournament_name, table="team_match")).drop_duplicates(subset="column")
    for row in available_measures.itertuples(index=False):
        label = str(row.label)
        options.append(label)
        mapping[label] = str(row.column)
    return options, mapping

def render_group(title: str, group: str) -> None:
    selected = registry.loc[registry["group"].eq(group)]
    prepared = []
    for measure in selected.itertuples(index=False):
        all_values = pd.to_numeric(all_team_means[measure.column], errors="coerce")
        team_values = pd.to_numeric(team_rows[measure.column], errors="coerce")
        spread = all_values.std(ddof=0)
        if team_values.notna().any() and all_values.notna().any():
            prepared.append(
                {
                    "measure": measure.label,
                    "value": float(team_values.mean()),
                    "average": float(all_values.mean()),
                    "z_score": float((team_values.mean() - all_values.mean()) / spread)
                    if spread > 0 else 0.0,
                    "unit": measure.unit,
                    "higher_is_better": measure.higher_is_better,
                }
            )
    if not prepared:
        return
    table = pd.DataFrame(prepared)
    st.subheader(title)
    figure = comparison_chart(table, "measure", "z_score", 0.0)
    figure.update_xaxes(title_text="Difference from tournament average (standard deviations)")
    st.plotly_chart(figure, width="stretch", key=f"{group}_comparison")
    st.caption(
        "Each blue bar compares the team's average with the tournament average; the horizontal axis is the distance "
        "from average in standard deviations (units of typical team-to-team spread), and the dashed line marks "
        "average. Use the table's direction column to see whether a higher or lower value is better for each measure."
    )
    show_table(
        table.rename(
            columns={
                "measure": "Measure",
                "value": "Team average",
                "average": "Tournament average",
                "z_score": "Distance from average",
                "unit": "Unit",
                "higher_is_better": "Higher value is better",
            }
        ).round(3),
        hide_index=True,
        width="stretch",
    )


render_group("Dominance", "dominance")
render_group("Cost of control and chances created", "efficiency")
render_group("Squad use and chance spread", "load")
render_group("How widely chances were created", "threat")

extra_groups = [
    ("chances", "Chances created"),
    ("contests", "Contests"),
    ("territory", "Territory"),
    ("resources", "Running and resources"),
    ("style", "Team shape and phases of play"),
    ("context", "Squad and match context"),
]
for group, title in extra_groups:
    all_group = measures(group=group, table="team_match")
    if not all_group.empty and all_group["tier"].eq("men").all():
        if tournament == "women":
            st.write("Men only: FIFA physical and broadcast tracking data.")
            continue
        st.caption("Men only: FIFA physical and broadcast tracking data.")
    selected = registry.loc[registry["group"].eq(group)]
    if selected.empty:
        continue
    if tournament == "women" and selected["tier"].eq("men").all():
        st.write("Men only: FIFA physical and broadcast tracking data.")
        continue
    render_group(title, group)

st.subheader("Results match by match")
performance = load("overperformance")
if performance is None:
    require("overperformance", "expectation")
else:
    trend = performance.loc[performance["tournament"].eq(tournament) & performance["team"].eq(team)].copy()
    extra = team_match.loc[team_match["tournament"].eq(tournament) & team_match["team"].eq(team)]
    trend = trend.merge(
        extra[["tournament", "match_id", "team", "dominance"]],
        on=["tournament", "match_id", "team"],
        how="left",
        validate="one_to_one",
    )
    trend = trend.sort_values(["match_date", "match_id"], ignore_index=True)
    st.plotly_chart(match_trend(trend), width="stretch")
    st.caption(
        "The blue line shows points above expectation and the grey line shows match control (the average share of "
        "contests, territory and chances). The horizontal axis is match date; higher blue values mean more points "
        "than expected, while higher grey values mean more control. Read each line over time, not one line's height "
        "against the other, because the measures use different scales."
    )

st.subheader("How the team changed through the tournament")
if performance is not None:
    options, measure_mapping = tournament_measure_options(tournament, team_match)
    default_index = options.index("Overall dominance") if "Overall dominance" in options else 0
    measure_label = st.selectbox("Choose a line", options, index=default_index, key="team_lines_measure")
    if measure_label == "Points above expectation":
        measure_column = measure_mapping[measure_label]
        label = "Points above expectation"
        plot_rows = team_match.loc[team_match["tournament"].eq(tournament), ["tournament", "match_id", "team", "match_number"]].copy()
        plot_rows = plot_rows.merge(
            performance.loc[performance["tournament"].eq(tournament), ["tournament", "match_id", "team", "overperformance"]],
            on=["tournament", "match_id", "team"],
            how="left",
            validate="many_to_one",
        )
    else:
        measure_column = measure_mapping[measure_label]
        label = measure_label
        plot_rows = team_match.loc[team_match["tournament"].eq(tournament), ["tournament", "match_id", "team", "match_number", measure_column]].copy()
    stage_map = stage_boundaries(tournament, team_match, performance)
    st.plotly_chart(tournament_lines(plot_rows, measure_column, label, team, stage_map), width="stretch")
    if measure_label == "Points above expectation":
        direction = "Values above zero mean more points than expected."
    else:
        measure_info = registry.loc[registry["column"].eq(measure_column)]
        higher_is_better = measure_info.iloc[0]["higher_is_better"] if not measure_info.empty else pd.NA
        direction = (
            "Higher values are better."
            if pd.notna(higher_is_better) and bool(higher_is_better)
            else "Lower values are better."
            if pd.notna(higher_is_better)
            else "The best direction depends on the measure."
        )
    st.caption(
        f"The selected team is blue and other teams are light grey; dotted vertical lines mark changes in stage. "
        f"The horizontal axis is match number and the vertical axis is {label}. {direction}"
    )
else:
    st.info("The tournament over-performance table is not available yet.")

zone_data = load("team_tournament_zone")
if zone_data is None:
    require("team_tournament_zone", "features")
else:
    zone_rows = zone_data.loc[zone_data["tournament"].eq(tournament) & zone_data["team"].eq(team)]
    st.subheader("Where the team had more territory or won more contests")
    zone_measure = st.radio(
        "Show on the pitch",
        ["Territory share", "Contests won share"],
        horizontal=True,
        key="team_zone_measure",
    )
    zone_map(zone_rows, "territory_share" if zone_measure == "Territory share" else "contests_share")
    st.caption(
        "Each tile is one pitch zone, arranged in the team's attacking direction from left to right. Blue means the "
        "team had the larger share of the selected measure there, orange means the opponent, and grey is near even. "
        "A larger team share means more territory or more contests won in that zone, which can help explain where "
        "match control came from."
    )

phase = load("team_tournament_phase")
if phase is None:
    require("team_tournament_phase", "features")
else:
    team_phase = phase.loc[phase["tournament"].eq(tournament) & phase["team"].eq(team)]
    tournament_phase = phase.loc[phase["tournament"].eq(tournament)]
    st.subheader("Chances and possession by phase")
    phase_column = st.radio(
        "Compare by phase",
        ["xG per possession", "Share of possessions"],
        horizontal=True,
        key="team_phase_metric",
    )
    phase_figure = phase_dot_plot(
        team_phase,
        tournament_phase,
        team,
        column="xg_per_possession" if phase_column == "xG per possession" else "share_of_possessions",
    )
    if phase_figure.data:
        st.plotly_chart(phase_figure, width="stretch")
        st.caption(
            "Each dot is one phase of play. Blue is this team and grey is the tournament average; the horizontal axis "
            "lists the phases and the vertical axis shows the selected measure. More xG per possession means more "
            "expected chance value each time the team had the ball; a larger possession share shows emphasis, not "
            "automatically better performance."
        )
    show_table(
        team_phase.rename(
            columns={
                "tournament": "Tournament",
                "team": "Team",
                "phase": "Phase of play",
                "possessions": "Possessions",
                "possession_seconds": "Time on the ball (seconds)",
                "shots": "Shots",
                "xg": "xG (expected goals)",
                "obv": "On-ball value (OBV)",
                "xg_per_possession": "Expected goals per possession",
                "share_of_possessions": "Share of possessions",
            }
        ),
        hide_index=True,
        width="stretch",
    )

for optional, stage_name in (("style_types", "styles"), ("efficiency_team_tournament", "features")):
    table = load(optional)
    if table is None:
        require(optional, stage_name)
    else:
        selected = table.loc[table["tournament"].eq(tournament) & table["team"].eq(team)]
        if not selected.empty:
            st.subheader(
                "Team style"
                if optional == "style_types"
                else "Efficiency index (combined cost and chance score)"
            )
            show_table(selected, hide_index=True, width="stretch")

st.subheader("Three things to notice")
findings = build_findings(team, team_match, tournament)
if findings:
    for finding in findings:
        st.write(f"- {finding}")
else:
    st.info("There are not enough built measures to prepare findings.")

export = team_rows.copy()
private_columns = [
    column
    for column in export
    if "person" in column.casefold() or "player" in column.casefold()
]
export = export.drop(columns=private_columns, errors="ignore")
st.download_button(
    "Download team-level numbers as CSV",
    export.to_csv(index=False).encode("utf-8"),
    file_name=f"{tournament}_{team.replace(' ', '_')}_summary.csv",
    mime="text/csv",
)
