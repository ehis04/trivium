from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from components.data import load
from components.layout import page_setup, show_table
from components.sidebar import select_context
from trivium.config import load_config
from trivium.db import tables
from trivium.definitions import POINT_DEFINITIONS, RESEARCH_QUESTION, STRENGTH
from trivium.measures import measures


page_setup("Method and limits")
tournament, _, _ = select_context()

st.subheader("The question")
st.write(RESEARCH_QUESTION)

st.subheader("How we measure team strength")
st.write(STRENGTH)

st.subheader("The three points")
for point, definition in POINT_DEFINITIONS.items():
    st.markdown(f"**{point}:** {definition}")

st.subheader("What the measures mean")
definitions = measures(tournament=tournament)
definitions = definitions[["label", "unit", "higher_is_better", "tier", "signifier", "definition"]].copy()
definitions = definitions.rename(columns={
    "label": "Measure", "unit": "Unit", "higher_is_better": "Higher is better",
    "tier": "Coverage", "signifier": "Project measure number", "definition": "Plain-English definition",
})
definitions["Coverage"] = definitions["Coverage"].map({"men": "Men only", "both": "Men and women"})
definitions["Higher is better"] = definitions["Higher is better"].map(
    {True: "Higher is better", False: "Lower is better"}
).fillna("Direction not defined")
show_table(definitions, hide_index=True, width="stretch")

st.subheader("Data and sources")
st.write(
    "- StatsBomb match events and scorecards show team actions, territory and chances.\n"
    "- Past international results provide the senior Elo rating (the team's strength estimate before the tournament).\n"
    "- FIFA camera-tracking and physical data support men's running measures; displays are by team or position group.\n"
    "- Prior men's under-17 tournament records provide background on teams' previous experience.\n"
    "- Providers can count actions differently; their figures are labelled separately."
)

st.subheader("Which data is available")
st.write(
    "Data coverage and known gaps across sources. "
    "The women's tournament uses StatsBomb events only (no camera tracking or FIFA physical measures)."
)

# 1. FIFA match folders and physical metrics
cf_folders = load("checks_match_folders")
st.markdown("#### FIFA match files and running measures (men)")
if cf_folders is None:
    st.info("not built in this build")
else:
    found_row = cf_folders[cf_folders["check"].str.contains("match folders found", case=False)]
    n_folders = int(found_row["value"].iloc[0]) if not found_row.empty else len(cf_folders)
    no_phys = cf_folders[
        cf_folders["check"].str.contains("no physical CSV", case=False)
        & ~cf_folders["check"].str.contains("folders with a physical CSV")
    ]
    gaps_phys = len(no_phys)
    st.write(f"- **Coverage**: FIFA distance and speed figures and match events (men).")
    st.write(f"- **Matches**: {n_folders} match folders found.")
    if gaps_phys > 0:
        st.write(f"- **Gaps**: {gaps_phys} folder{'s' if gaps_phys != 1 else ''} without a physical file.")
    else:
        st.write("- **Gaps**: None (all folders contain a physical file).")

# 2. Optical tracking
cf_track = load("checks_tracking")
st.markdown("#### Camera tracking (men)")
if cf_track is None:
    st.info("not built in this build")
else:
    track_zip_row = cf_track[cf_track["check"].str.contains("folders with a tracking zip", case=False)]
    zip_val = int(track_zip_row["value"].iloc[0]) if not track_zip_row.empty else "N/A"
    no_zip = cf_track[cf_track["check"].str.contains("no tracking zip", case=False)]
    missing_zip = 0
    if not track_zip_row.empty:
        target_str = str(track_zip_row["target"].iloc[0])
        if "of" in target_str:
            tot = int(target_str.split("of")[-1].strip())
            missing_zip = max(0, tot - int(track_zip_row["value"].iloc[0]))
    sync_checks = cf_track[cf_track["check"].str.contains("sync median error yd", case=False)]
    sync_gaps = int(sync_checks["passed"].eq(False).sum()) if not sync_checks.empty else 0
    imputed_checks = cf_track[cf_track["check"].str.contains("imputed player-frames share", case=False)]
    imputed_gaps = int(imputed_checks["passed"].eq(False).sum()) if not imputed_checks.empty else 0

    st.write("- **Coverage**: Camera recordings of player and ball positions, speeds and who had the ball (men).")
    st.write(f"- **Matches**: {zip_val} matches with tracking data.")
    gaps_track = []
    if missing_zip > 0 or len(no_zip) > 0:
        gaps_track.append(f"{max(missing_zip, len(no_zip))} match(es) without a tracking zip")
    if sync_gaps > 0:
        gaps_track.append(f"{sync_gaps} half/halves with tracking-to-event position differences over 5 yards")
    if imputed_gaps > 0:
        gaps_track.append(f"{imputed_gaps} match(es) with more than 15% of positions filled in by the provider")
    if gaps_track:
        st.write("- **Gaps**: " + "; ".join(gaps_track) + ".")
    else:
        st.write("- **Gaps**: None.")

# 3. Post-match technical reports
cf_rep = load("checks_reports")
st.markdown("#### Post-match team reports")
if cf_rep is None:
    st.info("not built in this build")
else:
    men_rep = cf_rep[cf_rep["check"].str.contains("men: reports read", case=False)]
    women_rep = cf_rep[cf_rep["check"].str.contains("women: reports read", case=False)]
    m_val = int(men_rep["value"].iloc[0]) if not men_rep.empty else 0
    w_val = int(women_rep["value"].iloc[0]) if not women_rep.empty and pd.notna(women_rep["value"].iloc[0]) else 0
    unread = cf_rep[cf_rep["check"].str.contains("report not read or not matched", case=False)]
    missing_figs = cf_rep[cf_rep["check"].str.contains("report figure missing", case=False)]

    st.write(f"- **Coverage**: Post-match technical reports for men and available public tournament PDFs for women.")
    st.write(f"- **Matches**: {m_val} men's matches, {w_val} women's matches read.")
    gaps_rep = []
    if not unread.empty:
        gaps_rep.append(f"{len(unread)} unread/unmatched report(s)")
    if not missing_figs.empty:
        gaps_rep.append(f"{len(missing_figs)} figure(s) missing from reports")
    if gaps_rep:
        st.write("- **Gaps**: " + "; ".join(gaps_rep) + ".")
    else:
        st.write("- **Gaps**: None.")

st.subheader("Checks on the data and analysis")
cfg = load_config()
check_names = [name for name in tables(cfg) if name.startswith("checks_")]
check_report = load("check_report")
check_tables = []
for name in check_names:
    table = load(name)
    if table is not None:
        table = table.copy()
        table["stage"] = name.removeprefix("checks_")
        check_tables.append(table)
if check_report is not None:
    check_tables.append(check_report.copy())
if check_tables:
    checks = pd.concat(check_tables, ignore_index=True, sort=False)
    if "passed" in checks:
        checks["passed"] = checks["passed"].map({True: "Passed", False: "Failed"})
        mixed_columns = {
            column: "string"
            for column in checks.columns
            if checks[column].dtype == object
        }
        checks = checks.astype(mixed_columns)
        summary = checks.groupby("stage")["passed"].agg(
            checks="size",
            passed=lambda values: int(values.eq("Passed").sum()),
            failed=lambda values: int(values.eq("Failed").sum()),
        ).reset_index()
        show_table(
            summary.rename(
                columns={
                    "stage": "Stage",
                    "checks": "Checks",
                    "passed": "Passed",
                    "failed": "Failed",
                }
            ),
            hide_index=True,
            width="stretch",
        )
    check_display = checks.rename(columns={
        "stage": "Part of the build", "check": "What was checked", "target": "Expected",
        "value": "Observed", "passed": "Result", "test": "Check number", "test_name": "Check purpose",
        "status": "Status", "measure": "Measure", "finding_type": "Finding",
        "label": "Name", "estimate": "Estimated difference", "low": "Lower end of range",
        "high": "Upper end of range", "wording": "Plain-language result",
    })
    show_table(check_display, hide_index=True, width="stretch")
else:
    st.info("No build check tables are available yet.")

reliability = load("reliability")
if reliability is None:
    st.info("Not built yet: needs table `reliability` from the `models` stage.")
else:
    st.subheader("Do the measures stay consistent?")
    selected_reliability = reliability.loc[reliability["tournament"].eq(tournament)]
    show_table(selected_reliability.rename(columns={
        "tournament": "Tournament", "measure": "Measure", "label": "Measure name", "tier": "Coverage",
        "n_teams": "Teams", "r_split": "Similarity between tournament halves",
        "r_sb": "Consistency score after adjustment",
        "r_sb_random_mean": "Average after shuffling matches", "passed": "Consistent enough",
    }), hide_index=True, width="stretch")

luck = load("luck_summary")
if luck is not None:
    selected_luck = luck.loc[luck["tournament"].eq(tournament)]
    if not selected_luck.empty:
        st.subheader("Teams with results outside the likely range")
        show_table(selected_luck.rename(columns={
            "tournament": "Tournament", "teams": "Teams", "above": "Above the likely range",
            "below": "Below the likely range", "expected_by_chance": "Number expected by chance",
        }), hide_index=True, width="stretch")
        st.caption(
            "The table compares how many teams finished above or below their likely result range with how many "
            "would be expected to do so through chance alone."
        )

st.subheader("Team profiles (combinations of the three scores)")
axes = load("tactical_profile_axes")
if axes is None:
    st.info("not built in this build")
else:
    rows = []
    for _, r in axes.iterrows():
        tourn = "Men" if r.get("tournament") == "men" else "Women"
        score_label = str(r.get("label", r.get("axis", "")))
        parts_str = str(r.get("parts", ""))

        r_val = r.get("r_sb")
        rep = r.get("repeatable")
        rep_text = (
            f"{r_val:.2f} ({'repeatable' if bool(rep) else 'not repeatable'})"
            if pd.notna(r_val)
            else "n/a"
        )

        sp = r.get("spearman")
        sp_low = r.get("spearman_low")
        sp_high = r.get("spearman_high")
        assoc_text = (
            f"{sp:+.2f} ({sp_low:+.2f} to {sp_high:+.2f})"
            if pd.notna(sp) and pd.notna(sp_low) and pd.notna(sp_high)
            else "n/a"
        )

        el = r.get("early_late_spearman")
        el_low = r.get("early_late_low")
        el_high = r.get("early_late_high")
        pred = r.get("predicts_later")
        pred_label = "predicts later" if bool(pred) else "does not clearly predict"
        oos_text = (
            f"{el:+.2f} ({el_low:+.2f} to {el_high:+.2f}; {pred_label})"
            if pd.notna(el) and pd.notna(el_low) and pd.notna(el_high)
            else "n/a"
        )

        rows.append(
            {
                "Tournament": tourn,
                "Profile score": score_label,
                "What the score includes": parts_str,
                "Consistency across matches": rep_text.replace("repeatable", "consistent"),
                "Link with points beyond strength prediction": assoc_text,
                "Early matches compared with later results": oos_text.replace(
                    "predicts later", "matches later results"
                ).replace("does not clearly predict", "does not clearly match"),
            }
        )
    profile_table = pd.DataFrame(rows)
    show_table(profile_table, hide_index=True, width="stretch")
    st.caption(
        "A consistency score closer to one means the measure behaves more similarly across matches; a positive "
        "link means higher scores tended to go with more points above expectation. The ranges show how uncertain "
        "those estimates are, and early-versus-later compares separate parts of each team's tournament."
    )

    st.markdown(
        "**Profile limits:**\n"
        "- **Few teams in each group**: There are eight possible profiles, so comparisons can be uncertain.\n"
        "- **Some measures vary by match**: A measure that changes a lot from one match to the next is less useful for planning.\n"
        "- **Different data coverage**: The women's tournament has fewer teams and StatsBomb event data only, with no camera tracking or FIFA physical data."
    )

st.subheader("Limits")
st.write(
    "- Each team plays a small number of matches, so averages and likely result ranges can be uncertain.\n"
    "- Men and women are analysed separately; camera tracking is available for men only.\n"
    "- A link between a measure and results does not prove one caused the other.\n"
    "- Synthetic demo values are fake and must never be treated as tournament results."
)
