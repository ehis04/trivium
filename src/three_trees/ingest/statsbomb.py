"""StatsBomb ingest for both tournaments.

Concatenates every event part file per tournament before any per-match work,
sets boolean dtypes explicitly, parses nested columns (tactics,
shot_freeze_frame, defensive_responsibility, related_events) with
ast.literal_eval. Period 5 (shootouts) goes to the shootouts table and is
excluded from all performance measures.

Output tables: events, matches, shootouts, checks_statsbomb.
"""

from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd

from three_trees.config import TOURNAMENTS, Config
from three_trees.storage import write_table
from three_trees.teams import clean_team_name

# Read as bool in some files and float in others (files with no TRUE values).
FLAG_COLUMNS: tuple[str, ...] = (
    "ball_recovery_offensive", "ball_recovery_recovery_failure", "block_deflection",
    "block_offensive", "block_save_block", "clearance_aerial_won", "clearance_head",
    "clearance_left_foot", "clearance_other", "clearance_right_foot", "counterpress",
    "dribble_no_touch", "dribble_nutmeg", "dribble_overrun", "foul_committed_advantage",
    "foul_committed_offensive", "foul_committed_penalty", "foul_won_advantage",
    "foul_won_defensive", "foul_won_penalty", "goalkeeper_punched_out",
    "goalkeeper_shot_saved_off_target", "goalkeeper_shot_saved_to_post",
    "goalkeeper_success_in_play", "goalkeeper_success_out", "half_start_late_video_start",
    "injury_stoppage_in_chain", "is_controlled_possession", "is_transition",
    "miscontrol_aerial_won", "off_camera", "out", "pass_aerial_won", "pass_backheel",
    "pass_cross", "pass_cut_back", "pass_deflected", "pass_goal_assist", "pass_inswinging",
    "pass_miscommunication", "pass_no_touch", "pass_outswinging", "pass_shot_assist",
    "pass_straight", "pass_switch", "pass_through_ball", "player_off_permanent",
    "shot_aerial_won", "shot_deflected", "shot_first_time", "shot_follows_dribble",
    "shot_kick_off", "shot_one_on_one", "shot_open_goal", "shot_redirect",
    "shot_saved_off_target", "shot_saved_to_post", "shot_shot_goal_assist",
    "shot_shot_shot_assist", "under_pressure",
)
TRUE_VALUES = frozenset({"true", "1", "1.0"})
INT_COLUMNS = ("match_id", "index", "period", "minute", "second")
NESTED_COLUMNS = ("tactics", "shot_freeze_frame", "defensive_responsibility", "related_events")
# Text lists "[x, y]" (shot_end_location also carries z), split into <col>_x and <col>_y.
LOCATION_COLUMNS = ("location", "pass_end_location", "carry_end_location", "shot_end_location")
SHOOTOUT_PERIOD = 5

MATCH_COLUMNS = (
    "tournament", "match_id", "match_date", "kick_off", "competition_stage",
    "home_team", "away_team", "home_team_id", "away_team_id", "home_score", "away_score",
    "neutral_ground", "shootout", "shootout_winner", "collection_status", "data_version",
)
SHOOTOUT_COLUMNS = (
    "tournament", "match_id", "team", "team_id", "kicks_taken", "kicks_scored", "won", "period5_events",
)

TARGETS = {
    "men": {"rows": 307_480, "matches": 104, "shootouts": 10},
    "women": {"rows": 159_256, "matches": 52, "shootouts": 6},
}
DATA_VERSION = "1.1.0"


def parse_literal(value: Any) -> Any:
    """Python-literal text (nested columns) -> object; empty -> None."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return None
    text = str(value).strip()
    return ast.literal_eval(text) if text else None


def to_flag(series: pd.Series) -> pd.Series:
    """True / 1.0 -> True; empty / 0 / False -> False."""
    return series.astype("string").str.strip().str.lower().isin(TRUE_VALUES).astype(bool)


def timestamp_seconds(series: pd.Series) -> pd.Series:
    """'HH:MM:SS.fff' (resets each period) -> seconds within the period."""
    parts = series.astype("string").str.split(":", expand=True)
    return parts[0].astype(float) * 3600 + parts[1].astype(float) * 60 + parts[2].astype(float)


def split_location(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'[x, y]' or '[x, y, z]' text -> numeric x and y."""
    parts = series.astype("string").str.strip("[] ").str.split(",", expand=True)
    x = pd.to_numeric(parts[0], errors="coerce") if 0 in parts else pd.Series(np.nan, index=series.index)
    y = pd.to_numeric(parts[1], errors="coerce") if 1 in parts else pd.Series(np.nan, index=series.index)
    return x.astype(float), y.astype(float)


def read_event_files(paths: list) -> pd.DataFrame:
    """Every part file, in the given order, concatenated before any per-match work."""
    if not paths:
        raise FileNotFoundError("No StatsBomb event files found.")
    dtypes = {c: object for c in FLAG_COLUMNS} | {"id": "string"}
    parts = [pd.read_csv(p, dtype=dtypes, low_memory=False) for p in paths]
    return pd.concat(parts, ignore_index=True)


def prepare_events(raw: pd.DataFrame, tournament: str) -> pd.DataFrame:
    """Explicit dtypes, clean team name, period_seconds and split locations."""
    ev = raw.copy()
    for col in FLAG_COLUMNS:
        if col in ev:
            ev[col] = to_flag(ev[col])
    for col in INT_COLUMNS:
        ev[col] = ev[col].astype("int64")
    ev["id"] = ev["id"].astype("string")
    ev["team_id"] = ev["team_id"].astype("int64")
    ev["team_name"] = ev["team"]
    ev["team"] = ev["team_name"].map(clean_team_name)
    ev.insert(0, "tournament", tournament)
    ev["period_seconds"] = timestamp_seconds(ev["timestamp"])
    for col in LOCATION_COLUMNS:
        if col in ev:
            ev[f"{col}_x"], ev[f"{col}_y"] = split_location(ev[col])
    return ev.sort_values(["match_id", "index"], kind="stable").reset_index(drop=True)


def load_events(cfg: Config, tournament: str) -> pd.DataFrame:
    """All events of the tournament, every period (shootouts split off later)."""
    return prepare_events(read_event_files(cfg.statsbomb[tournament].event_files()), tournament)


def _name_of(value: Any) -> Any:
    """Plain text stays as it is; dict text like "{'id': 1, 'name': 'Goal'}" -> 'Goal'."""
    if isinstance(value, str) and value.strip().startswith("{"):
        parsed = parse_literal(value)
        return parsed.get("name") if isinstance(parsed, dict) else value
    return value


def split_shootouts(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(events without period 5, one row per team per shootout match).

    period5_events counts every period 5 row of the team (not only kicks), so the
    row-count check can add them back.
    """
    is_shootout = events["period"] == SHOOTOUT_PERIOD
    p5 = events[is_shootout]
    keys = ["tournament", "match_id", "team", "team_id"]
    shots = p5[p5["type"].map(_name_of) == "Shot"]
    scored = shots.assign(scored=shots["shot_outcome"].map(_name_of) == "Goal")
    shootouts = (
        p5.groupby(keys).size().rename("period5_events").to_frame()
        .join(scored.groupby(keys)["scored"].agg(kicks_taken="size", kicks_scored="sum"))
        .fillna(0)
        .astype("int64")
        .reset_index()
    )
    by_match = shootouts.groupby(["tournament", "match_id"])["kicks_scored"]
    best = by_match.transform("max")
    on_best = by_match.transform(lambda s: (s == s.max()).sum())
    shootouts["won"] = (shootouts["kicks_scored"] == best) & (on_best == 1)
    return events[~is_shootout].reset_index(drop=True), shootouts[list(SHOOTOUT_COLUMNS)]


def prepare_matches(raw: pd.DataFrame, tournament: str, shootouts: pd.DataFrame | None = None) -> pd.DataFrame:
    """Matches table (columns in MATCH_COLUMNS); scores exclude shootout kicks.

    Every other matches-file column (managers and their dates of birth, referees,
    stadiums) is dropped here.
    """
    m = raw.copy()
    m["tournament"] = tournament
    m["match_date"] = pd.to_datetime(m["match_date"]).dt.date
    m["competition_stage"] = m["competition_stage"].map(_name_of)
    for side in ("home", "away"):
        m[f"{side}_team"] = m[f"{side}_team"].map(_name_of).map(clean_team_name)
    m["neutral_ground"] = to_flag(m["neutral_ground"])
    for col in ("match_id", "home_team_id", "away_team_id", "home_score", "away_score"):
        m[col] = m[col].astype("int64")
    for col in ("kick_off", "competition_stage", "collection_status", "data_version"):
        m[col] = m[col].astype("string")
    winners = {}
    if shootouts is not None and len(shootouts):
        so = shootouts[shootouts["tournament"] == tournament]
        winners = dict(zip(so.loc[so["won"], "match_id"], so.loc[so["won"], "team"]))
        m["shootout"] = m["match_id"].isin(set(so["match_id"]))
    else:
        m["shootout"] = False
    m["shootout_winner"] = m["match_id"].map(winners).astype("string").fillna("")
    return m[list(MATCH_COLUMNS)].sort_values(["match_date", "kick_off", "match_id"]).reset_index(drop=True)


def load_matches(cfg: Config, tournament: str, shootouts: pd.DataFrame | None = None) -> pd.DataFrame:
    return prepare_matches(pd.read_csv(cfg.statsbomb[tournament].matches), tournament, shootouts)


def goals_by_team(events: pd.DataFrame) -> pd.DataFrame:
    """Goals in periods 1-2 per match and team id: Shot outcome Goal + Own Goal For (benefiting team)."""
    ev = events[events["period"].isin([1, 2])]
    goal = ((ev["type"] == "Shot") & (ev["shot_outcome"] == "Goal")) | (ev["type"] == "Own Goal For")
    return (
        ev[goal].groupby(["tournament", "match_id", "team_id"]).size().rename("goals").reset_index()
    )


def _check(rows: list, check: str, target: Any, value: Any, passed: bool) -> None:
    rows.append({"check": check, "target": str(target), "value": str(value), "passed": bool(passed)})


def check_statsbomb(events: pd.DataFrame, shootouts: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Data checks. `events` holds every period, shootout events included."""
    rows: list[dict] = []
    for t in sorted(matches["tournament"].unique()):
        target = TARGETS.get(t, {})
        ev = events[events["tournament"] == t]
        m = matches[matches["tournament"] == t]
        so = shootouts[shootouts["tournament"] == t]
        n_matches = m["match_id"].nunique()

        _check(rows, f"{t}: rows (events + shootout events)", f"{target.get('rows', 0):,}",
               f"{len(ev):,}", len(ev) == target.get("rows"))
        _check(rows, f"{t}: matches", target.get("matches"), n_matches, n_matches == target.get("matches"))

        ev_ids, m_ids = set(ev["match_id"]), set(m["match_id"])
        _check(rows, f"{t}: every events match_id in matches and vice versa", "yes",
               f"{len(ev_ids - m_ids)} events-only, {len(m_ids - ev_ids)} matches-only", ev_ids == m_ids)

        dup = int(ev["id"].duplicated().sum())
        _check(rows, f"{t}: duplicate event ids", 0, dup, dup == 0)

        contiguous = ev.groupby("match_id")["index"].apply(
            lambda s: bool(np.array_equal(np.sort(s.to_numpy()), np.arange(1, len(s) + 1)))
        )
        _check(rows, f"{t}: index contiguous from 1 in every match (all periods)",
               f"{n_matches} of {n_matches}", f"{int(contiguous.sum())} of {n_matches}",
               int(contiguous.sum()) == n_matches)

        goals = goals_by_team(ev).set_index(["match_id", "team_id"])["goals"]
        home = m.set_index(["match_id", "home_team_id"]).index.map(lambda k: goals.get(k, 0))
        away = m.set_index(["match_id", "away_team_id"]).index.map(lambda k: goals.get(k, 0))
        ok = int(((np.asarray(home) == m["home_score"].to_numpy()) & (np.asarray(away) == m["away_score"].to_numpy())).sum())
        _check(rows, f"{t}: goals in periods 1-2 = home_score, away_score",
               f"{n_matches} of {n_matches}", f"{ok} of {n_matches}", ok == n_matches)

        periods = sorted(int(p) for p in ev["period"].unique())
        _check(rows, f"{t}: periods present", "{1, 2, 5}", set(periods), set(periods) == {1, 2, 5})

        n_so = so["match_id"].nunique()
        _check(rows, f"{t}: shootout matches", target.get("shootouts"), n_so, n_so == target.get("shootouts"))
        one_winner = int(so.groupby("match_id")["won"].sum().eq(1).sum())
        _check(rows, f"{t}: shootout matches with one winner", f"{n_so} of {n_so}",
               f"{one_winner} of {n_so}", one_winner == n_so)

        complete = int(((m["collection_status"] == "Complete") & (m["data_version"] == DATA_VERSION)).sum())
        _check(rows, f"{t}: collection_status Complete, data_version {DATA_VERSION}",
               f"{n_matches} of {n_matches}", f"{complete} of {n_matches}", complete == n_matches)
    return pd.DataFrame(rows, columns=["check", "target", "value", "passed"])

def run(cfg: Config) -> None:
    all_events, events, shootouts, matches = [], [], [], []
    for t in TOURNAMENTS:
        ev = load_events(cfg, t)
        kept, so = split_shootouts(ev)
        all_events.append(ev[["tournament", "match_id", "id", "index", "period", "type", "team_id", "shot_outcome"]])
        events.append(kept)
        shootouts.append(so)
        matches.append(load_matches(cfg, t, so))
        print(f"  {t}: {len(ev):,} events, {len(kept):,} kept, {so['match_id'].nunique()} shootouts", flush=True)

    shootouts_df = pd.concat(shootouts, ignore_index=True)
    matches_df = pd.concat(matches, ignore_index=True)
    checks = check_statsbomb(pd.concat(all_events, ignore_index=True), shootouts_df, matches_df)

    write_table(cfg, "events", pd.concat(events, ignore_index=True))
    write_table(cfg, "matches", matches_df)
    write_table(cfg, "shootouts", shootouts_df)
    write_table(cfg, "checks_statsbomb", checks)
    with pd.option_context("display.width", 200, "display.max_colwidth", 70):
        print(checks.to_string(index=False))
