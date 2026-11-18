from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from components.data import load, require
from components.layout import page_setup
from three_trees.charts.answer import point_evidence_chart
from three_trees.charts.simple import evidence_bars, scatter_fit
from three_trees.charts.tactical import profile_outcomes
from three_trees.definitions import POINTS, POINT_DEFINITIONS, PROFILE_SCORES, RESEARCH_QUESTION, STRENGTH
from three_trees.models.answer import point_evidence, point_verdicts


def _text_list(value: object) -> str:
    return str(value) if pd.notna(value) and str(value).strip() else "none identified"


def _score_sentence(axes: pd.DataFrame | None, axis: str, score_label: str, tournament: str) -> str:
    if axes is None or axes.empty:
        return f"The {score_label} score has no result built for the {tournament} tournament."
    match = axes.loc[axes["tournament"].eq(tournament) & axes["axis"].eq(axis)]
    if match.empty:
        return f"The {score_label} score has no result built for the {tournament} tournament."
    row = match.iloc[0]
    rho = pd.to_numeric(pd.Series([row.get("spearman")]), errors="coerce").iloc[0]
    low = pd.to_numeric(pd.Series([row.get("spearman_low")]), errors="coerce").iloc[0]
    high = pd.to_numeric(pd.Series([row.get("spearman_high")]), errors="coerce").iloc[0]
    if pd.notna(rho) and pd.notna(low) and pd.notna(high):
        association = f"its rank correlation was {rho:+.2f} (95% interval {low:+.2f} to {high:+.2f})"
    else:
        association = "its rank correlation is not available"
    repeatable = (
        "was repeatable" if pd.notna(row.get("repeatable")) and bool(row["repeatable"])
        else "was not repeatable" if pd.notna(row.get("repeatable"))
        else "has no repeatability result"
    )
    predicts_later = (
        "clearly predicted later matches" if pd.notna(row.get("predicts_later")) and bool(row["predicts_later"])
        else "did not clearly predict later matches" if pd.notna(row.get("predicts_later"))
        else "has no later-match result"
    )
    return f"For {tournament}, the {score_label} score: {association}; it {repeatable} and {predicts_later}."


def _gate_status(value: object) -> str:
    if pd.isna(value):
        return "not tested"
    return "passed" if bool(value) else "did not pass"


page_setup("The research question")
st.write(RESEARCH_QUESTION)
st.write(STRENGTH)

comparison = load("profile_comparison")
if comparison is None:
    require("profile_comparison", "profiles")
    evidence = pd.DataFrame()
    verdicts = pd.DataFrame()
else:
    evidence = point_evidence(comparison)
    verdicts = point_verdicts(comparison)

axes = load("tactical_profile_axes")
profile_summary = load("tactical_profile_summary")
finding_gate = load("finding_gate")
sensitivity = load("tactical_profile_sensitivity")
team_tournament = load("team_tournament")
team_summary = load("team_summary")
# The measure that carries each point, shown against points above expectation with a line of best fit.
KEY_MEASURES = {
    "Dominance": ("chances_share", "Share of the match's chances (xG share)", "share of the chances"),
    "Resourcefulness": ("defensive_work_per_dominance", "Defensive actions per point of dominance (cost of control)",
                        "cost of control"),
    "Efficiency": ("security", "xG conceded per opponent entry into the defensive third", "xG conceded per entry"),
}
pattern = None
if team_tournament is not None and team_summary is not None:
    pattern = team_tournament.merge(team_summary[["tournament", "team", "overperformance_per_match"]],
                                    on=["tournament", "team"], how="inner")

st.subheader("The short answer")
if verdicts.empty:
    st.info("The three points cannot be compared until the profile evidence is built.")
else:
    for row in verdicts.itertuples(index=False):
        st.markdown(
            f"**{row.point} — {row.verdict}.** "
            f"Men: clear differences in {_text_list(row.men_clear)}; no clear difference in "
            f"{_text_list(row.men_not_clear)}. Women: clear differences in {_text_list(row.women_clear)}; "
            f"no clear difference in {_text_list(row.women_not_clear)}."
        )

for point in POINTS:
    st.subheader(point)
    st.write(POINT_DEFINITIONS[point])
    if evidence.empty:
        st.info("No point evidence is available yet.")
    else:
        for column, tournament in zip(st.columns(2), ("men", "women")):
            with column:
                st.markdown(f"**{'Men' if tournament == 'men' else 'Women'}**")
                st.plotly_chart(evidence_bars(evidence, tournament, point), width="stretch",
                                key=f"simple_evidence_{point}_{tournament}")
                st.caption(
                    "Simple view. One bar per measure: how much higher (right) or lower (left) teams that beat "
                    "their strength were than the rest, after accounting for strength (1 = a large difference). "
                    "Green or orange bars are clear differences; grey bars are too small to tell from chance."
                )
        measure, axis_label, short = KEY_MEASURES[point]
        if pattern is not None and measure in pattern:
            for column, tournament in zip(st.columns(2), ("men", "women")):
                rows = pattern.loc[pattern["tournament"].eq(tournament)]
                figure, fit = scatter_fit(rows, measure, "overperformance_per_match", axis_label,
                                          "Points above expectation per match")
                with column:
                    st.plotly_chart(figure, width="stretch", key=f"simple_fit_{point}_{tournament}")
                    if pd.notna(fit["slope"]):
                        st.caption(
                            f"Each dot is a {'men' if tournament == 'men' else 'women'}'s team; the straight line is "
                            f"the line of best fit. It {'rises' if fit['slope'] > 0 else 'falls'}: teams with a higher "
                            f"{short} won {'more' if fit['slope'] > 0 else 'fewer'} points above what their strength "
                            f"predicted (correlation {fit['r']:+.2f}, {int(fit['n'])} teams; 0 means no pattern)."
                        )
        st.plotly_chart(point_evidence_chart(evidence, point), width="stretch", key=f"point_evidence_{point}")
        st.caption(
            "Each row is a measure grouped by part. Blue is the men's tournament and pink is the women's. "
            "Dots compare teams above with teams below their Elo expectation; bars show the 95% interval. "
            "Filled dots have intervals clear of zero, while hollow dots include zero. Right of zero means "
            "higher among teams beating their strength; left means lower."
        )
    verdict_row = verdicts.loc[verdicts["point"].eq(point)]
    if verdict_row.empty:
        st.write("The evidence verdict for this point is not available.")
    else:
        st.write(f"The evidence verdict for this point is **{verdict_row.iloc[0]['verdict']}**.")
    score_key, (_, score_label, _) = next(
        (key, value) for key, value in PROFILE_SCORES.items() if value[0] == point
    )
    for tournament in ("men", "women"):
        st.write(_score_sentence(axes, score_key, score_label, tournament))

st.subheader("The profile that combines them")
if profile_summary is None:
    require("tactical_profile_summary", "profiles")
else:
    left, right = st.columns(2)
    for column, tournament in ((left, "men"), (right, "women")):
        with column:
            st.markdown(f"**{tournament.capitalize()}**")
            st.plotly_chart(
                profile_outcomes(profile_summary, tournament),
                width="stretch",
                key=f"profile_outcomes_{tournament}",
            )
            st.caption(
                "Each dot is a team profile's mean points above Elo expectation per match; bars show the "
                "95% interval and the dashed line is zero. Green is above expectation, orange is below; "
                "a positive value is higher than strength predicted."
            )

if finding_gate is None:
    require("finding_gate", "validation")
    passing_findings = pd.DataFrame()
else:
    passing_findings = finding_gate.loc[
        finding_gate["finding_type"].eq("tactical profile")
        & finding_gate["passes_gate"].fillna(False).astype(bool)
    ]
if passing_findings.empty:
    st.write("No tactical profile wording currently passes the evidence gate.")
else:
    st.write("Evidence-gated wording:")
    for wording in passing_findings["wording"].dropna():
        st.write(f"- {wording}")

st.subheader("How sure")
if sensitivity is None or sensitivity.empty:
    st.info("Sensitivity and profile stability checks are not built yet.")
else:
    for (tournament, score), rows in sensitivity.groupby(["tournament", "score"], sort=True):
        same_sign = rows["same_sign_as_main"].fillna(False).astype(bool)
        changed = rows.loc[~same_sign, "variant"].astype(str).tolist()
        if changed:
            stability_text = f"the sign changed under {', '.join(changed)}"
        else:
            stability_text = "the sign stayed the same under every tested variant"
        st.write(
            f"For the {tournament} {score} score, {stability_text} "
            f"({int(same_sign.sum())} of {len(rows)} variants kept the main sign)."
        )
    stability_columns = ["tournament", "median_threshold_share", "leave_one_out_share"]
    if set(stability_columns).issubset(sensitivity.columns):
        stability = sensitivity[stability_columns].drop_duplicates("tournament")
        for row in stability.itertuples(index=False):
            median_share = pd.to_numeric(pd.Series([row.median_threshold_share]), errors="coerce").iloc[0]
            loo_share = pd.to_numeric(pd.Series([row.leave_one_out_share]), errors="coerce").iloc[0]
            median_text = f"{median_share:.0%}" if pd.notna(median_share) else "not available"
            loo_text = f"{loo_share:.0%}" if pd.notna(loo_share) else "not available"
            st.write(
                f"For the {row.tournament} tournament, {median_text} of team labels matched after using "
                f"median score cut-offs, and {loo_text} matched when one team at a time was left out."
            )

tactical_gate = (
    finding_gate.loc[finding_gate["finding_type"].eq("tactical profile")]
    if finding_gate is not None and "finding_type" in finding_gate
    else pd.DataFrame()
)
if tactical_gate.empty:
    st.write("The tactical-profile evidence gate has not been built.")
else:
    for row in tactical_gate.itertuples(index=False):
        checks = (
            f"repeatability {_gate_status(row.test3_repeatable)}, "
            f"strength adjustment {_gate_status(row.test4_not_just_strength)}, "
            f"later-match test {_gate_status(row.test5_out_of_sample)}, and "
            f"95% interval {_gate_status(row.test6_interval_excludes_zero)}"
        )
        st.write(f"For {row.tournament}, the {row.label} gate {_gate_status(row.passes_gate)}: {checks}.")

st.subheader("What we cannot say")
st.write("These results show associations, not causes.")
st.write("A result described as “no clear difference” is not proof that there is no effect.")
st.write("The women's tournament is smaller, so its estimates are less precise.")
unclear_points = (
    verdicts.loc[verdicts["verdict"].eq("no clear difference"), "point"].tolist()
    if not verdicts.empty else []
)
if unclear_points:
    st.write(f"No clear difference was found for these points overall: {', '.join(unclear_points)}.")
if not verdicts.empty:
    for row in verdicts.itertuples(index=False):
        for tournament, unclear in (("men", row.men_not_clear), ("women", row.women_not_clear)):
            if _text_list(unclear) != "none identified":
                st.write(f"For {row.point}, no clear difference in the {tournament} tournament for {unclear}.")
unclear_scores: list[str] = []
if axes is not None and not axes.empty:
    for row in axes.itertuples(index=False):
        reasons = []
        if not bool(getattr(row, "spearman_excludes_zero", False)):
            reasons.append("the rank-correlation interval includes zero")
        if pd.isna(row.repeatable) or not bool(row.repeatable):
            reasons.append("the score is not repeatable")
        if pd.isna(row.predicts_later) or not bool(row.predicts_later):
            reasons.append("it does not clearly predict later matches")
        if reasons:
            unclear_scores.append(f"{row.tournament} {row.label} ({'; '.join(reasons)})")
if unclear_scores:
    st.write(f"Scores without clear supporting evidence: {', '.join(unclear_scores)}.")
