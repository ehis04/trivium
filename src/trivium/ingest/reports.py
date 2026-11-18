"""FIFA post-match report ingest (men's from the match folders, women's from
the public reports folder) with pdfplumber: key stats (page 3) and phases
of play (page 4).

Each key-stat line of the report reads "<home value> <label> <away value>" and each
phases-of-play line "<home %> <phase> <away %>". The two team names are read from the
phases page header ("Qatar Phases of Play Italy") and paired with the StatsBomb names of the
match. Men's reports are matched to a StatsBomb match through their folder; women's through
the date in the page header and the two team names. Where a folder holds an original and a
"V2" report, the V2 is read. Only these two pages are read; the reports' player pages are not.

Output tables: report_team_match, checks_reports.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
import pdfplumber

from trivium.config import Config
from trivium.storage import read_table, table_exists, write_table

KEY_STATS_TITLE = "Match Summary - Key Statistics"
PHASES_TITLE = "Phases of Play"
NUM = r"(\d+(?:\.\d+)?)"

# Key-stat lines: column(s) for the home team, then the same for the away team.
KEY_STATS = [
    (["goals"], rf"^{NUM} Goals {NUM}$"),
    (["attempts", "attempts_on_target"], rf"^{NUM} \({NUM}\) Attempts at Goal \(On Target\) {NUM} \({NUM}\)$"),
    (["passes", "passes_complete"], rf"^{NUM} \({NUM}\) Total Passes \(Complete\) {NUM} \({NUM}\)$"),
    (["pass_completion_pct"], rf"^{NUM} ?% Pass Completion % {NUM} ?%$"),
    (["line_breaks_completed"], rf"^{NUM} Completed Line Breaks {NUM}$"),
    (["defensive_line_breaks"], rf"^{NUM} Defensive Line Breaks {NUM}$"),
    (["receptions_final_third"], rf"^{NUM} Receptions in the Final Third {NUM}$"),
    (["crosses"], rf"^{NUM} Crosses {NUM}$"),
    (["ball_progressions"], rf"^{NUM} Ball Progressions {NUM}$"),
    (["pressures_applied", "direct_pressures"],
     rf"^{NUM} \({NUM}\) Defensive Pressures Applied \(Direct Pressures\) {NUM} \({NUM}\)$"),
    (["forced_turnovers"], rf"^{NUM} Forced Turnovers {NUM}$"),
    (["second_balls"], rf"^{NUM} Second Balls {NUM}$"),
    (["distance_km"], rf"^{NUM} km Total Distance Covered {NUM} km$"),
    (["zone4_km"], rf"^{NUM} km Zone 4\b.* {NUM} km$"),
]
POSSESSION = rf"^Total {NUM}% {NUM}% {NUM}% Total$"  # home, in contest, away

# Phases of play page: label -> column name.
PHASES = {
    "Build Up Unopposed": "phase_build_up_unopposed_pct", "Build Up Opposed": "phase_build_up_opposed_pct",
    "Progression": "phase_progression_pct", "Final Third": "phase_final_third_pct",
    "Long Ball": "phase_long_ball_pct", "Attacking Transition": "phase_attacking_transition_pct",
    "Counter Attack": "phase_counter_attack_pct", "Set Piece": "phase_set_piece_pct",
    "High Press": "phase_high_press_pct", "Mid Press": "phase_mid_press_pct", "Low Press": "phase_low_press_pct",
    "High Block": "phase_high_block_pct", "Mid Block": "phase_mid_block_pct", "Low Block": "phase_low_block_pct",
    "Recovery": "phase_recovery_pct", "Defensive Transition": "phase_defensive_transition_pct",
    "Counter-press": "phase_counter_press_pct",
}
STAT_COLUMNS = ["possession_pct", "possession_in_contest_pct",
                *[c for cols, _ in KEY_STATS for c in cols], *PHASES.values()]
MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August",
                                      "September", "October", "November", "December"], start=1)}
DATE = re.compile(r"(\d{1,2}) (" + "|".join(MONTHS) + r") (\d{4})")


def parse_key_stats(text: str) -> dict[str, dict[str, float]]:
    """{"home": {...}, "away": {...}} from the key statistics page text; a stat not found is left out."""
    sides: dict[str, dict[str, float]] = {"home": {}, "away": {}}
    for line in (ln.strip() for ln in text.splitlines()):
        if m := re.match(POSSESSION, line):
            sides["home"]["possession_pct"], contest, sides["away"]["possession_pct"] = map(float, m.groups())
            sides["home"]["possession_in_contest_pct"] = sides["away"]["possession_in_contest_pct"] = contest
            continue
        for cols, pattern in KEY_STATS:
            if m := re.match(pattern, line):
                values = list(map(float, m.groups()))
                sides["home"].update(zip(cols, values[: len(cols)]))
                sides["away"].update(zip(cols, values[len(cols):]))
                break
    return sides


def parse_phases(text: str) -> tuple[str | None, str | None, dict[str, dict[str, float]]]:
    """(home name, away name, {"home": {...}, "away": {...}}) from the phases of play page text."""
    home = away = None
    sides: dict[str, dict[str, float]] = {"home": {}, "away": {}}
    for line in (ln.strip() for ln in text.splitlines()):
        if PHASES_TITLE in line and home is None:
            home, _, away = (part.strip() for part in line.partition(PHASES_TITLE))
            continue
        for label, col in PHASES.items():
            if m := re.match(rf"^{NUM}% {re.escape(label)} {NUM}%$", line):
                sides["home"][col], sides["away"][col] = map(float, m.groups())
                break
    return home or None, away or None, sides


def read_report(path: str | Path) -> dict[str, object]:
    """Both teams' report figures from one post-match report PDF (pages found by their titles)."""
    logging.getLogger("pdfminer").setLevel(logging.ERROR)  # font warnings in these PDFs, not errors
    stats = phases = header = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages[:8]:
            text = page.extract_text() or ""
            if KEY_STATS_TITLE in text and not stats:
                stats, header = text, text.splitlines()[0]
            elif PHASES_TITLE in text and not phases:
                phases = text
    home, away, phase_values = parse_phases(phases)
    key = parse_key_stats(stats)
    date = DATE.search(header)
    return {
        "home": home, "away": away,
        "date": pd.Timestamp(int(date.group(3)), MONTHS[date.group(2)], int(date.group(1))).date() if date else None,
        "values": {side: {**key[side], **phase_values[side]} for side in ("home", "away")},
    }


def report_rows(report: dict[str, object], match_id: int, statsbomb: list[str], tournament: str) -> pd.DataFrame:
    """One row per team for one report, team = the StatsBomb name (paired with the report's names)."""
    from trivium.ingest.match_folders import pair_team_names

    names = [n for n in (report["home"], report["away"]) if n]
    pairs = pair_team_names(names, statsbomb)
    rows = []
    for side in ("home", "away"):
        rows.append({"tournament": tournament, "match_id": match_id, "team": pairs.get(report[side]),
                     "report_team": report[side], **report["values"][side]})
    return pd.DataFrame(rows).reindex(columns=["tournament", "match_id", "team", "report_team", *STAT_COLUMNS])


def report_path(folder: Path) -> Path | None:
    """The folder's post-match report, preferring a "V2" when both are there."""
    pdfs = sorted(folder.glob("*POST MATCH REPORT*.pdf"))
    return next((p for p in pdfs if "V2" in p.stem.upper()), pdfs[0] if pdfs else None)


def run(cfg: Config) -> None:
    """reports stage: report_team_match from every men's folder and any women's reports, and checks_reports."""
    from trivium.ingest.match_folders import match_folders, pair_team_names

    matches = read_table(cfg, "matches") if table_exists(cfg, "matches") else None
    checks: list[dict] = []

    def add(check: str, target: str, value: object, passed: bool | None) -> None:
        checks.append({"check": check, "target": target, "value": value,
                       "passed": None if passed is None else bool(passed)})

    parts, unread = [], []
    # Men: each match folder's report, matched through the folder.
    folders = match_folders(cfg)
    men = matches[matches["tournament"] == "men"] if matches is not None else None
    for f in folders[folders["matched"]].itertuples():
        path = report_path(cfg.match_folders_men / f.folder)
        if path is None:
            unread.append(f.folder)
            continue
        statsbomb = list(men.loc[men["match_id"] == f.match_id, ["home_team", "away_team"]].iloc[0]) \
            if men is not None else [f.home_team, f.away_team]
        parts.append(report_rows(read_report(path), int(f.match_id), statsbomb, "men"))

    # Women: public reports, matched on the date in the page header and the two team names.
    women_pdfs = sorted(cfg.fifa_reports_women.glob("*.pdf")) if cfg.fifa_reports_women.is_dir() else []
    women = matches[matches["tournament"] == "women"] if matches is not None else None
    for path in women_pdfs:
        report = read_report(path)
        found = None
        if women is not None and report["date"] is not None:
            day = women[pd.to_datetime(women["match_date"]).dt.date == report["date"]]
            for r in day.itertuples():
                if len(pair_team_names([report["home"], report["away"]], [r.home_team, r.away_team])) == 2:
                    found = r
        if found is None:
            unread.append(path.name)
            continue
        parts.append(report_rows(report, int(found.match_id), [found.home_team, found.away_team], "women"))

    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["tournament", "match_id", "team"])
    write_table(cfg, "report_team_match", table.drop(columns="report_team", errors="ignore"))

    # Checks: every report read in full, teams paired, goals equal the matches table, possession adds up.
    n_men = int(folders["matched"].sum())
    men_rows = table[table["tournament"] == "men"]
    add("men: reports read", f"{n_men}", men_rows["match_id"].nunique(), men_rows["match_id"].nunique() == n_men)
    add("women: reports read", "52 (if the public PDFs are in fifa_reports_women)",
        int(table.loc[table["tournament"] == "women", "match_id"].nunique()), None if not women_pdfs else
        table.loc[table["tournament"] == "women", "match_id"].nunique() == 52)
    for name in unread:
        add(f"report not read or not matched: {name}", "read", 0, False)
    complete = table[STAT_COLUMNS].notna().all(axis=1) if len(table) else pd.Series(dtype=bool)
    add("team rows with every report figure", f"{len(table)}", int(complete.sum()), bool(complete.all()))
    for col in STAT_COLUMNS:
        missing = int(table[col].isna().sum()) if col in table else len(table)
        if missing:
            add(f"report figure missing: {col}", "0 rows", missing, False)
    add("team rows paired with a StatsBomb team", f"{len(table)}", int(table["team"].notna().sum()),
        bool(table["team"].notna().all()))
    if matches is not None and len(table):
        scores = pd.concat([
            matches[["tournament", "match_id", "home_team", "home_score"]].set_axis(["tournament", "match_id", "team", "score"], axis=1),
            matches[["tournament", "match_id", "away_team", "away_score"]].set_axis(["tournament", "match_id", "team", "score"], axis=1),
        ])
        goals = table.merge(scores, on=["tournament", "match_id", "team"], how="left")
        for t, g in goals.groupby("tournament"):
            same = (g["goals"] == g["score"]).groupby(g["match_id"]).all()
            add(f"{t}: report goals = StatsBomb score", f"{len(same)} of {len(same)}", int(same.sum()), bool(same.all()))
    pairs = table.groupby("match_id")["possession_pct"].sum() + table.groupby("match_id")["possession_in_contest_pct"].first()
    near = (pairs - 100).abs() <= 2
    add("possession: home + in contest + away within 2 of 100", f"{len(pairs)} of {len(pairs)}", int(near.sum()),
        bool(near.all()))

    checks_table = pd.DataFrame(checks, columns=["check", "target", "value", "passed"])
    write_table(cfg, "checks_reports", checks_table)
    with pd.option_context("display.width", 200, "display.max_colwidth", 80, "display.max_rows", None):
        print(checks_table.to_string(index=False))
