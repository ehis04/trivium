"""Provider match folder ingest (men only, 104 folders).

Reads the physical metrics CSV, FIFA unified event data and Enhanced phase
file per match. Players are joined to StatsBomb on team + shirt number with
a name check fallback; unmatched rows are listed.

Output tables: physical_player_match, fifa_events, checks_match_folders (player_lookup and phase_clips are
written by the tracking stage, which has the tracking person ids).
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import orjson
import pandas as pd

from trivium.config import Config
from trivium.storage import read_table, table_exists, write_table
from trivium.teams import _aliases, clean_team_name
from trivium.tracking.stream import read_metadata

# FIFA physical metrics CSV (build context 4.2 a): file column -> table column.
# "Player Name" is left out on purpose, so names never enter any table.
PHYSICAL_COLUMNS = {
    "Match ID": "fifa_match_id",
    "Team Name": "fifa_team_name",  # upper case in the file, e.g. "ITALY"
    "Player ID": "fifa_player_id",
    "Jersey #": "jersey_number",
    "Total Duration (min)": "total_duration_min",
    "Total Distance (m)": "total_distance_m",
    "1-7 km/h (m)": "dist_1_7_kmh_m",
    "7-15 km/h (m)": "dist_7_15_kmh_m",
    "15-20 km/h (m)": "dist_15_20_kmh_m",
    "20-25 km/h (m)": "dist_20_25_kmh_m",
    "25+ km/h (m)": "dist_25_plus_kmh_m",
    "Max Speed (km/h)": "max_speed_kmh",
    "# Speed Runs": "speed_runs",
    "# Sprints": "sprints",
}


def read_physical(path: str | Path) -> pd.DataFrame:
    """One FIFA physical metrics CSV: one row per player, snake_case columns, no player names."""
    # Read only the columns we keep; header spaces are trimmed first in case of stray spaces.
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    missing = [c for c in PHYSICAL_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"{Path(path).name} is missing columns: {', '.join(missing)}")
    return df[list(PHYSICAL_COLUMNS)].rename(columns=PHYSICAL_COLUMNS)


# ---------------------------------------------------------------------------
# Match folder -> StatsBomb match_id (Task 3.5)
# ---------------------------------------------------------------------------

# File-name prefix, e.g. "20251103_FU17WC_01_QAT-ITA_": the first 8 digits are the local match date.
PREFIX = re.compile(r"^(\d{8})_FU17WC_")
TRACKING_SUFFIX = "_RSV_Broadcast_Tracking_Data_Tactical_Wide.zip"
PHYSICAL_SUFFIX = "_FIFA_Player_Physical_Metrics_Data.csv"
ENHANCED_SUFFIX = "_FIFA_Enhanced.json"


EVENTS_METADATA_SUFFIX = "_FIFA_Unified_Event_Data_Metadata.json"


def _plain(name: str) -> str:
    # Accents dropped, case and extra spaces ignored: "CÔTE D'IVOIRE" -> "cote d'ivoire".
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_name.casefold().split())


def team_key(name: pd.Series) -> pd.Series:
    """Team names for comparing across providers: age-group suffix removed, case and accents
    ignored, then the senior-results name from lookup/countries.csv. So "USA" = "United States",
    "KOREA DPR" = "North Korea" and "COTE D'IVOIRE" = "Côte d'Ivoire U17"."""
    aliases = {_plain(k): _plain(v) for k, v in _aliases().items()}
    plain = name.fillna("").map(clean_team_name).map(_plain)
    return plain.map(lambda n: aliases.get(n, n))


def pair_team_names(names: list[str], statsbomb: list[str]) -> dict[str, str]:
    """Each provider's team name in one match -> the StatsBomb name of the same team.

    Names are matched through team_key; when exactly one name is left on each side,
    they are the match's other team ("UAE" -> "United Arab Emirates" with no alias needed).
    """
    keys = dict(zip(team_key(pd.Series(statsbomb)), statsbomb))
    pairs = {n: keys[k] for n, k in zip(names, team_key(pd.Series(names))) if k in keys}
    left_over = [n for n in names if n not in pairs]
    unused = [s for s in statsbomb if s not in pairs.values()]
    if len(left_over) == 1 and len(unused) == 1:
        pairs[left_over[0]] = unused[0]
    return pairs


def folder_info(folder: Path) -> dict[str, object]:
    """Match date (from the file-name prefix) and both team names, from the tracking metadata
    (home_team, away_team) and from FIFA's events metadata (fifa_home_team, fifa_away_team)."""
    names = [p.name for p in folder.iterdir() if p.is_file()]
    dates = sorted({m.group(1) for n in names if (m := PREFIX.match(n))})
    tracking = [folder / n for n in names if n.endswith(TRACKING_SUFFIX)]
    events_meta = [folder / n for n in names if n.endswith(EVENTS_METADATA_SUFFIX)]
    meta = read_metadata(tracking[0]) if tracking else {}
    fifa = orjson.loads(events_meta[0].read_bytes()) if events_meta else {}
    return {
        "folder": folder.name,
        "match_date": pd.to_datetime(dates[0]).date() if len(dates) == 1 else None,  # None if missing or mixed
        "home_team": meta.get("HomeTeam", {}).get("Name"),
        "away_team": meta.get("AwayTeam", {}).get("Name"),
        "fifa_home_team": fifa.get("home_team_name"),
        "fifa_away_team": fifa.get("away_team_name"),
    }


def map_folders(folders: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """StatsBomb match_id for each folder: same date and the same two teams (in either order).

    folders: folder, match_date and team names from folder_info. The tracking names are tried
    first, then FIFA's events-metadata names for folders still without a match.
    matches: match_id, match_date, home_team, away_team (the matches table).
    Returns folders with match_id added (empty when no match) and matched True/False.
    """
    def pairs(home: pd.Series, away: pd.Series) -> list[tuple[str, str]]:
        # The two teams as one sorted pair, so home/away order does not matter.
        return [tuple(sorted(t)) for t in zip(team_key(home), team_key(away))]

    # (date, team pair) -> StatsBomb match_id
    match_dates = pd.to_datetime(matches["match_date"]).dt.date
    lookup = dict(zip(zip(match_dates, pairs(matches["home_team"], matches["away_team"])), matches["match_id"]))

    out = folders.copy()
    out["match_id"] = pd.array([pd.NA] * len(out), dtype="Int64")
    dates = pd.to_datetime(out["match_date"]).dt.date
    for home, away in (("home_team", "away_team"), ("fifa_home_team", "fifa_away_team")):
        if home in out:
            found = [lookup.get(key) for key in zip(dates, pairs(out[home], out[away]))]
            out["match_id"] = out["match_id"].fillna(pd.Series(found, index=out.index, dtype="Int64"))
    out["matched"] = out["match_id"].notna()
    return out


def men_matches(cfg: Config) -> pd.DataFrame:
    """Men's matches: match_id, match_date, home_team, away_team.

    From the matches table (built by the statsbomb stage); until that table exists,
    the same four columns are read from the StatsBomb matches CSV.
    """
    columns = ["match_id", "match_date", "home_team", "away_team"]
    if table_exists(cfg, "matches"):
        matches = read_table(cfg, "matches")
        return matches.loc[matches["tournament"] == "men", columns]
    return pd.read_csv(cfg.statsbomb["men"].matches, usecols=columns)


def match_folders(cfg: Config) -> pd.DataFrame:
    """Every provider match folder with its StatsBomb match_id and the paths of its tracking zip,
    physical CSV and Enhanced file (empty where the file is not there), in match order."""
    folders_dir = cfg.match_folders_men
    paths = sorted(
        (p for p in folders_dir.iterdir() if p.is_dir()) if folders_dir.is_dir() else [],
        key=lambda p: int(m.group(1)) if (m := re.match(r"M(\d+)", p.name)) else 10**6,
    )
    columns = ["folder", "match_date", "home_team", "away_team", "match_id", "matched", "tracking_zip", "physical_csv",
               "enhanced_json", "folder_path"]
    if not paths:
        return pd.DataFrame(columns=columns)

    def first(folder: Path, suffix: str) -> Path | None:
        return next(iter(sorted(folder.glob(f"*{suffix}"))), None)

    folders = pd.DataFrame([folder_info(p) for p in paths])
    folders = map_folders(folders, men_matches(cfg))
    folders["tracking_zip"] = [first(p, TRACKING_SUFFIX) for p in paths]
    folders["physical_csv"] = [first(p, PHYSICAL_SUFFIX) for p in paths]
    folders["enhanced_json"] = [first(p, ENHANCED_SUFFIX) for p in paths]
    folders["folder_path"] = paths
    return folders[columns]


def physical_player_match(folders: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """FIFA physical rows for every matched folder that has a physical CSV.

    folders: from match_folders. matches: match_id, home_team, away_team.
    Adds tournament, match_id and team: the clean StatsBomb name of the FIFA team
    ("ITALY" -> "Italy"), empty when the FIFA name is neither team in that match.
    """
    # For each match, the two StatsBomb team names keyed by their comparison form.
    names = {
        r.match_id: {team_key(pd.Series([t])).iloc[0]: clean_team_name(t) for t in (r.home_team, r.away_team)}
        for r in matches.itertuples()
    }
    parts = []
    for f in folders[folders["matched"] & folders["physical_csv"].notna()].itertuples():
        df = read_physical(f.physical_csv)
        df.insert(0, "team", team_key(df["fifa_team_name"]).map(names[f.match_id]))
        df.insert(0, "match_id", int(f.match_id))
        df.insert(0, "tournament", "men")
        parts.append(df)
    columns = ["tournament", "match_id", "team", *PHYSICAL_COLUMNS.values()]
    return pd.concat(parts, ignore_index=True)[columns] if parts else pd.DataFrame(columns=columns)


# ---------------------------------------------------------------------------
# FIFA unified events (round 2, Task 3.7)
# ---------------------------------------------------------------------------

FIFA_EVENTS_SUFFIX = "_FIFA_Unified_Event_Data_Events.csv"
FIFA_EVENT_COLUMNS = [
    "category", "event_type", "event", "outcome", "game_period", "match_run_time_in_ms", "pressure",
    "line_break_outcome", "line_break_direction", "team_units_broken", "total_team_units", "team_shape",
    "team_unit", "sequence_type", "x_location_start", "y_location_start", "x_location_end", "y_location_end",
    "from_player_id", "from_player_shirt_number", "to_player_id", "to_player_shirt_number",
]
FIFA_HALVES = {1: "first_half", 2: "second_half", 3: "extra_time_first_half", 4: "extra_time_second_half"}
REFEREES = "TEAM REFEREES"


def read_fifa_events(path: str | Path, statsbomb: list[str]) -> pd.DataFrame:
    """One FIFA unified events CSV: team (StatsBomb name) + FIFA_EVENT_COLUMNS, no player names.

    game_period comes from half_time (the file's own game_period is filled only on period markers);
    referee rows and shootout rows (half_time 5) are left out.
    """
    df = pd.read_csv(path, low_memory=False)
    df = df[(df["team_name"] != REFEREES) & df["half_time"].isin(list(FIFA_HALVES))].copy()
    pairs = pair_team_names(sorted(df["team_name"].unique()), statsbomb)
    df["team"] = df["team_name"].map(pairs)
    df["game_period"] = df["half_time"].map(FIFA_HALVES)
    return df[["team", *FIFA_EVENT_COLUMNS]]


def fifa_events(folders: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """fifa_events for every matched folder with a FIFA events file."""
    statsbomb = {int(r.match_id): [r.home_team, r.away_team] for r in matches.itertuples()}
    parts = []
    for f in folders[folders["matched"]].itertuples():
        path = next(iter(sorted(Path(f.folder_path).glob(f"*{FIFA_EVENTS_SUFFIX}"))), None)
        if path is not None:
            df = read_fifa_events(path, statsbomb[int(f.match_id)])
            df.insert(0, "match_id", int(f.match_id))
            df.insert(0, "tournament", "men")
            parts.append(df)
    columns = ["tournament", "match_id", "team", *FIFA_EVENT_COLUMNS]
    return pd.concat(parts, ignore_index=True)[columns] if parts else pd.DataFrame(columns=columns)


# ---------------------------------------------------------------------------
# Player lookup: StatsBomb <-> FIFA <-> tracking ids (round 2, Task 3.6; internal, never shown)
# ---------------------------------------------------------------------------

LOOKUP_COLUMNS = ["tournament", "match_id", "team", "statsbomb_player_id", "fifa_player_id", "tracking_person_id",
                  "jersey_number", "position_group", "method"]


def statsbomb_position_group(position: object) -> str:
    """Position group from a StatsBomb position name: Goalkeeper; Back -> Defender; Midfield -> Midfielder;
    Wing, Forward, Striker -> Forward."""
    name = str(position or "")
    if "Goalkeeper" in name:
        return "Goalkeeper"
    if "Back" in name:
        return "Defender"
    if "Midfield" in name:
        return "Midfielder"
    if any(w in name for w in ("Wing", "Forward", "Striker")):
        return "Forward"
    return "Unknown"


def statsbomb_lineups(events: pd.DataFrame) -> pd.DataFrame:
    """StatsBomb players of each team-match: match_id, team, statsbomb_player_id, jersey_number, position,
    and name_check (for the name check only; dropped before anything is written).

    From the Starting XI and Tactical Shift line-ups (with shirt numbers), then substitutes from
    Substitution events (no shirt number; position = that of the player replaced).
    """
    from ast import literal_eval

    rows = []
    lineups = events[events["type"].isin(["Starting XI", "Tactical Shift"]) & events["tactics"].notna()]
    for e in lineups.itertuples():
        tactics = literal_eval(e.tactics) if isinstance(e.tactics, str) else e.tactics
        for p in tactics.get("lineup", []):
            rows.append({"match_id": int(e.match_id), "team": e.team, "statsbomb_player_id": int(p["player"]["id"]),
                         "jersey_number": int(p["jersey_number"]), "position": p["position"]["name"],
                         "name_check": p["player"]["name"]})
    if "substitution_replacement_id" in events:
        subs = events[(events["type"] == "Substitution") & events["substitution_replacement_id"].notna()]
        for e in subs.itertuples():
            rows.append({"match_id": int(e.match_id), "team": e.team,
                         "statsbomb_player_id": int(e.substitution_replacement_id), "jersey_number": None,
                         "position": e.position, "name_check": e.substitution_replacement})
    out = pd.DataFrame(rows, columns=["match_id", "team", "statsbomb_player_id", "jersey_number", "position", "name_check"])
    out["jersey_number"] = pd.to_numeric(out["jersey_number"]).astype("Int64")
    return out.drop_duplicates(["match_id", "team", "statsbomb_player_id"], keep="first")


def _tokens(name: object) -> set[str]:
    return {t for t in _plain(str(name or "")).replace("-", " ").split() if len(t) >= 3}


def player_lookup_match(sb: pd.DataFrame, fifa: pd.DataFrame, tracking: pd.DataFrame) -> pd.DataFrame:
    """Player lookup for one match.

    sb: statsbomb_lineups rows (team, statsbomb_player_id, jersey_number, position, name_check).
    fifa: FIFA physical rows (team, fifa_player_id, jersey_number, name_check).
    tracking: tracked players (team, tracking_person_id, jersey_number, position_group, optional name_check).
    All teams are StatsBomb names. FIFA and tracking share shirt numbers, so they are joined on team + shirt.
    StatsBomb is joined on team + shirt ("shirt number"); players still unpaired are paired within the team by
    a name word of 3+ letters in common with one candidate only ("name check"); then, where a team has exactly
    one unpaired goalkeeper on each side, those two ("goalkeeper"); then, where a team has exactly one unpaired
    player on each side, those two ("only one left": both providers list the players who took part). The rest
    are "unmatched". Names (FIFA's, else the tracking team sheet's) are used only here.
    """
    ft = fifa.merge(tracking, on=["team", "jersey_number"], how="outer", suffixes=("", "_tracking"),
                    validate="one_to_one").reset_index(drop=True)
    if "name_check_tracking" in ft:  # FIFA's name where there is one, else the tracking team sheet's
        ft["name_check"] = ft["name_check"].combine_first(ft.pop("name_check_tracking"))
    sb_shirt = sb[sb["jersey_number"].notna()]
    by_shirt = sb_shirt.merge(ft, on=["team", "jersey_number"], how="outer", suffixes=("_sb", ""),
                              indicator=True, validate="one_to_one")
    matched = by_shirt[by_shirt["_merge"] == "both"].assign(method="shirt number")
    sb_left = pd.concat([by_shirt[by_shirt["_merge"] == "left_only"],
                         sb[sb["jersey_number"].isna()].rename(columns={"name_check": "name_check_sb"})],
                        ignore_index=True)
    ft_left = by_shirt[by_shirt["_merge"] == "right_only"]

    def is_goalkeeper(row: pd.Series) -> bool:
        return row.get("position_group") == "Goalkeeper" or statsbomb_position_group(row.get("position")) == "Goalkeeper"

    pairs, used_sb, used_ft = [], set(), set()

    def pair(s_idx: int, f_idx: int, method: str) -> None:
        s_row, f_row = sb_left.loc[s_idx], ft_left.loc[f_idx]
        used_sb.add(s_idx)
        used_ft.add(f_idx)
        pairs.append({**f_row.to_dict(), "statsbomb_player_id": s_row["statsbomb_player_id"],
                      "position": s_row["position"], "method": method})

    for s_idx, s_row in sb_left.iterrows():
        candidates = [f_idx for f_idx, f_row in ft_left[ft_left["team"] == s_row["team"]].iterrows()
                      if f_idx not in used_ft and _tokens(s_row["name_check_sb"]) & _tokens(f_row["name_check"])]
        if len(candidates) == 1:
            pair(s_idx, candidates[0], "name check")
    for team in sb_left["team"].unique():
        keepers_sb = [i for i, r in sb_left[sb_left["team"] == team].iterrows() if i not in used_sb and is_goalkeeper(r)]
        keepers_ft = [i for i, r in ft_left[ft_left["team"] == team].iterrows() if i not in used_ft and is_goalkeeper(r)]
        if len(keepers_sb) == 1 and len(keepers_ft) == 1:
            pair(keepers_sb[0], keepers_ft[0], "goalkeeper")
    for team in sb_left["team"].unique():
        last_sb = [i for i in sb_left[sb_left["team"] == team].index if i not in used_sb]
        last_ft = [i for i in ft_left[ft_left["team"] == team].index if i not in used_ft]
        if len(last_sb) == 1 and len(last_ft) == 1:
            pair(last_sb[0], last_ft[0], "only one left")

    rest = pd.concat([sb_left[~sb_left.index.isin(used_sb)], ft_left[~ft_left.index.isin(used_ft)]]).assign(
        method="unmatched")
    out = pd.concat([matched, pd.DataFrame(pairs), rest], ignore_index=True)
    for col in ("statsbomb_player_id", "fifa_player_id", "tracking_person_id", "jersey_number"):
        out[col] = pd.to_numeric(out.get(col), errors="coerce").astype("Int64")
    group = out["position_group"] if "position_group" in out else pd.Series(index=out.index, dtype=object)
    out["position_group"] = group.where(group.notna() & (group != "Unknown"), out["position"].map(statsbomb_position_group))
    return out.reindex(columns=[c for c in LOOKUP_COLUMNS if c not in ("tournament", "match_id")])


def read_physical_names(path: str | Path) -> pd.DataFrame:
    """fifa_player_id and name_check from a physical CSV, for the lookup's name check only (never written)."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df[["Player ID", "Player Name"]].set_axis(["fifa_player_id", "name_check"], axis=1)


def run(cfg: Config) -> None:
    """match_folders stage: physical_player_match and fifa_events from every folder, and checks_match_folders."""
    folders = match_folders(cfg)
    matches = men_matches(cfg)
    physical = physical_player_match(folders, matches)
    write_table(cfg, "physical_player_match", physical)
    events = fifa_events(folders, matches)
    write_table(cfg, "fifa_events", events)

    checks = []

    def add(check: str, target: str, value: float, passed: bool) -> None:
        checks.append({"check": check, "target": target, "value": float(value), "passed": bool(passed)})

    # Folders: found, matched to StatsBomb, with a physical CSV (each missing one listed).
    n = len(folders)
    add("match folders found", "104", n, n == 104)
    add("folders matched to a StatsBomb match_id", f"{n} of {n}", folders["matched"].sum(), folders["matched"].all())
    add("folders with a physical CSV", f"{n} of {n}", folders["physical_csv"].notna().sum(),
        folders["physical_csv"].notna().all())
    for f in folders[~folders["matched"]].itertuples():
        add(f"{f.folder}: not matched to StatsBomb", "matched", 0, False)
    for f in folders[folders["physical_csv"].isna()].itertuples():
        add(f"{f.folder}: no physical CSV", "present", 0, False)

    # Rows: per match, team names found in the StatsBomb match, one row per team + shirt number.
    per_match = physical.groupby("match_id").size()
    add("physical rows", "-", len(physical), len(physical) > 0)
    add("rows per match, fewest", "22 or more (two starting XIs)", per_match.min() if len(per_match) else 0,
        len(per_match) > 0 and per_match.min() >= 22)
    add("rows per match, most", "-", per_match.max() if len(per_match) else 0, True)
    add("rows with a StatsBomb team name", f"{len(physical)} of {len(physical)}",
        physical["team"].notna().sum(), physical["team"].notna().all())
    for name in sorted(physical.loc[physical["team"].isna(), "fifa_team_name"].unique()):
        add(f"FIFA team name not found in its match: {name}", "found", 0, False)
    dupes = physical.duplicated(["match_id", "team", "jersey_number"]).sum()
    add("duplicate team + shirt number in a match", "0", dupes, dupes == 0)
    add("teams", "48", physical["team"].nunique(), physical["team"].nunique() == 48)

    # FIFA events: every folder read, teams paired with StatsBomb, no shootout or referee rows.
    add("fifa_events: matches", f"{n}", events["match_id"].nunique(), events["match_id"].nunique() == n)
    add("fifa_events: rows", "-", len(events), len(events) > 0)
    add("fifa_events: rows with a StatsBomb team name", f"{len(events)}", events["team"].notna().sum(),
        bool(events["team"].notna().all()))
    add("fifa_events: matches with both teams", f"{n}", (events.groupby("match_id")["team"].nunique() == 2).sum(),
        bool((events.groupby("match_id")["team"].nunique() == 2).all()))

    from trivium.tracking.stream import with_tournament

    table = with_tournament(pd.DataFrame(checks, columns=["check", "target", "value", "passed"]))
    write_table(cfg, "checks_match_folders", table)
    with pd.option_context("display.width", 200, "display.max_colwidth", 70):
        print(table.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
