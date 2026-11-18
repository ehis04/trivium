"""StatsBomb senior open data adapter (test 10: does the method hold in the professional game?).

Downloads the matches and events files of four senior tournaments from StatsBomb open
data (matches/<competition_id>/<season_id>.json, events/<match_id>.json) into
<external.dir>/open_data/, skipping files already there. Each event is flattened to the
provider CSV forms: type, team, player position, possession team and play pattern as plain
names (team_id, player_id, possession_team_id beside them); event detail objects as
<object>_<key> columns with names as plain text (recipient and replacement ids beside
them); 50_50 and tactics kept as Python-literal dict text; location and end locations as
"[x, y]" text. The result goes through ingest.statsbomb.prepare_events, so the feature
code reads it exactly as it reads the provider tables.

Kept: the columns the feature code reads (SENIOR_EVENT_COLUMNS). Open data has no OBV,
HOPS, location bucket, possession directness or phases-of-play fields; those columns are
present and empty. Player names are dropped (lineups keep ids, positions and numbers).
Team names lose the " Women's" suffix and the "WNT " prefix. Periods 1-4 are kept (senior extra time);
period 5 goes to the shootout columns of senior_matches. The feature code reads
periods 1-2 only.

neutral_ground is False when a team's country is the stadium's country.
collection_status holds open data's match_status.

Output tables: senior_events, senior_matches, checks_open_data.
"""

from __future__ import annotations

import json
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from three_trees.config import Config
from three_trees.ingest import statsbomb as sb
from three_trees.storage import write_table
from three_trees.teams import clean_team_name

COMPETITIONS: dict[str, tuple[int, int]] = {
    "senior_men_wc2022": (43, 106),
    "senior_women_wc2023": (72, 107),
    "senior_men_euro2024": (55, 282),
    "senior_women_euro2025": (53, 315),
}
EXPECTED_MATCHES = {"senior_men_wc2022": 64, "senior_women_wc2023": 64, "senior_men_euro2024": 51,
                    "senior_women_euro2025": 31}
NAMED = ("type", "team", "player", "position", "possession_team", "play_pattern")
WITH_ID = frozenset({"team", "player", "possession_team"})
NESTED_DETAIL_IDS = frozenset({"recipient", "replacement"})
KEPT_AS_TEXT = ("50_50", "tactics")
WOMEN_SUFFIX = " Women's"
WOMEN_PREFIX = "WNT "  # e.g. "WNT Finland" in Women's Euro 2025
GOAL_PERIODS = (1, 2, 3, 4)
DOWNLOAD_WORKERS = 8
TIMEOUT_S = 120
RETRIES = 3
SENIOR_EVENT_COLUMNS = (
    "tournament", "match_id", "id", "index", "period", "minute", "second", "period_seconds",
    "type", "team", "team_id", "player_id", "position", "possession", "possession_team_id", "play_pattern",
    "location_x", "location_y", "location_bucket", "pass_end_location_x", "carry_end_location_x",
    "pass_outcome", "pass_shot_assist", "pass_goal_assist", "pass_assisted_shot_id",
    "pass_aerial_won", "pass_duel_win_probability",
    "shot_type", "shot_outcome", "shot_statsbomb_xg", "shot_key_pass_id", "shot_aerial_won",
    "shot_duel_win_probability",
    "duel_type", "duel_outcome", "duel_duel_win_probability", "50_50", "related_events",
    "ball_recovery_recovery_failure", "clearance_aerial_won", "clearance_duel_win_probability",
    "miscontrol_aerial_won", "miscontrol_duel_win_probability",
    "tactics", "substitution_replacement_id", "foul_committed_card", "bad_behaviour_card",
    "player_off_permanent",
    "possession_directness", "control_degree", "is_transition", "is_controlled_possession", "obv_total_net",
)
RAW_ONLY = ("timestamp", "location", "pass_end_location", "carry_end_location", "shot_end_location")


def open_data_dir(cfg: Config) -> Path:
    return cfg.external_dir / "open_data"


def download(base_url: str, rel: str, root: Path, force: bool = False) -> Path:
    """<base_url>/<rel> saved as <root>/<rel>; an existing file is kept unless force. A response
    that is not complete JSON (connection closed early) is fetched again, up to RETRIES times."""
    path = root / rel
    if force or not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        for attempt in range(RETRIES):
            with urllib.request.urlopen(f"{base_url.rstrip('/')}/{rel}", timeout=TIMEOUT_S) as response, \
                    tmp.open("wb") as f:
                shutil.copyfileobj(response, f)
            try:
                json.loads(tmp.read_bytes())
                break
            except json.JSONDecodeError:
                if attempt == RETRIES - 1:
                    raise
        tmp.replace(path)
    return path


def read_json(base_url: str, rel: str, root: Path) -> Any:
    """The parsed file; a saved file that does not parse is downloaded again."""
    try:
        return json.loads(download(base_url, rel, root).read_bytes())
    except json.JSONDecodeError:
        return json.loads(download(base_url, rel, root, force=True).read_bytes())


def fetch(cfg: Config, competition_id: int, season_id: int) -> tuple[list[dict], list[int]]:
    """(matches JSON, match ids); downloads the events files that are missing."""
    base, root = cfg.external_urls["statsbomb_open_data"], open_data_dir(cfg)
    matches = read_json(base, f"matches/{competition_id}/{season_id}.json", root)
    ids = [int(m["match_id"]) for m in matches]
    with ThreadPoolExecutor(DOWNLOAD_WORKERS) as pool:
        list(pool.map(lambda mid: download(base, f"events/{mid}.json", root), ids))
    return matches, ids


def _text(value: Any) -> Any:
    return str(value) if isinstance(value, (list, dict)) else value


def _strip_names(tactics: dict) -> dict:
    lineup = [{"player": {"id": p["player"]["id"]}, "position": p.get("position"),
               "jersey_number": p.get("jersey_number")} for p in tactics.get("lineup", [])]
    return {"formation": tactics.get("formation"), "lineup": lineup}


def flatten_event(event: dict) -> dict:
    """One open-data event -> one row in the provider CSV column names and value forms."""
    row: dict[str, Any] = {}
    for key, value in event.items():
        if key in NAMED:
            row[key] = value.get("name") if isinstance(value, dict) else value
            if key in WITH_ID and isinstance(value, dict):
                row[f"{key}_id"] = value.get("id")
        elif key == "tactics":
            row[key] = str(_strip_names(value))
        elif key in KEPT_AS_TEXT:
            row[key] = str(value)
        elif isinstance(value, dict):
            for sub, v in value.items():
                if sub == "freeze_frame":
                    continue
                name = f"{key}_{sub}"
                if isinstance(v, dict) and "name" in v:
                    row[name] = v["name"]
                    if sub in NESTED_DETAIL_IDS:
                        row[f"{name}_id"] = v.get("id")
                else:
                    row[name] = _text(v)
        else:
            row[key] = _text(value)
    return row


def _team(name: str) -> str:
    return clean_team_name(name.removesuffix(WOMEN_SUFFIX).removeprefix(WOMEN_PREFIX))


def events_frame(match_id: int, events: list[dict]) -> pd.DataFrame:
    """One match's flattened events in the provider raw column forms: every kept column, no others."""
    raw = pd.DataFrame([flatten_event(e) | {"match_id": match_id} for e in events])
    raw["team"] = raw["team"].map(lambda v: _team(v) if isinstance(v, str) else v)
    derived = {"tournament", "period_seconds"} | {f"{c}_{a}" for c in sb.LOCATION_COLUMNS for a in ("x", "y")}
    keep = [c for c in (*SENIOR_EVENT_COLUMNS, *RAW_ONLY) if c not in derived]
    return raw.reindex(columns=keep)


def prepare_senior_events(raw: pd.DataFrame, tournament: str) -> pd.DataFrame:
    """Every period, the kept columns only, in the forms of the provider events table."""
    ev = sb.prepare_events(raw, tournament)
    for col in ("player_id", "substitution_replacement_id", "possession_team_id"):
        ev[col] = pd.to_numeric(ev[col], errors="coerce")
    ev["possession_team_id"] = ev["possession_team_id"].astype("int64")
    ev["possession"] = ev["possession"].astype("int64")
    ev["shot_statsbomb_xg"] = pd.to_numeric(ev["shot_statsbomb_xg"], errors="coerce")
    return ev[list(SENIOR_EVENT_COLUMNS)]


def _country(side: dict) -> Any:
    return (side.get("country") or {}).get("name")


def raw_matches(matches: list[dict]) -> pd.DataFrame:
    """Open-data matches in the provider matches-file columns read by statsbomb.prepare_matches."""
    rows = []
    for m in matches:
        home, away = m["home_team"], m["away_team"]
        venue = ((m.get("stadium") or {}).get("country") or {}).get("name")
        hosted = venue is not None and venue in (_country(home), _country(away))
        rows.append({
            "match_id": m["match_id"], "match_date": m["match_date"], "kick_off": m.get("kick_off"),
            "competition_stage": (m.get("competition_stage") or {}).get("name"),
            "home_team": _team(home["home_team_name"]), "away_team": _team(away["away_team_name"]),
            "home_team_id": home["home_team_id"], "away_team_id": away["away_team_id"],
            "home_score": m["home_score"], "away_score": m["away_score"],
            "neutral_ground": not hosted, "collection_status": m.get("match_status"),
            "data_version": (m.get("metadata") or {}).get("data_version"),
        })
    return pd.DataFrame(rows)


def goals_all_periods(events: pd.DataFrame) -> pd.DataFrame:
    """Goals in periods 1-4 per match and team id: Shot outcome Goal + Own Goal For."""
    ev = events[events["period"].isin(GOAL_PERIODS)]
    goal = ((ev["type"] == "Shot") & (ev["shot_outcome"] == "Goal")) | (ev["type"] == "Own Goal For")
    return ev[goal].groupby(["tournament", "match_id", "team_id"]).size().rename("goals").reset_index()


def check_open_data(events: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """events: every period, shootouts included."""
    rows: list[dict] = []
    for t in sorted(matches["tournament"].unique()):
        m = matches[matches["tournament"] == t]
        ev = events[events["tournament"] == t]
        n = len(m)
        expected = EXPECTED_MATCHES.get(t)
        rows.append({"check": f"{t}: matches", "target": str(expected), "value": str(n),
                     "passed": n == expected})
        with_events = int(m["match_id"].isin(set(ev["match_id"])).sum())
        rows.append({"check": f"{t}: matches with events", "target": f"{n} of {n}",
                     "value": f"{with_events} of {n}", "passed": with_events == n})
        goals = goals_all_periods(ev).set_index(["match_id", "team_id"])["goals"]
        home = [goals.get(k, 0) for k in zip(m["match_id"], m["home_team_id"])]
        away = [goals.get(k, 0) for k in zip(m["match_id"], m["away_team_id"])]
        ok = int(((np.asarray(home) == m["home_score"].to_numpy())
                  & (np.asarray(away) == m["away_score"].to_numpy())).sum())
        rows.append({"check": f"{t}: goals in periods 1-4 = home_score, away_score (extra time included)",
                     "target": f"{n} of {n}", "value": f"{ok} of {n}", "passed": ok == n})
        periods = sorted(int(p) for p in ev["period"].unique())
        rows.append({"check": f"{t}: periods present",
                     "target": "1-2, extra time 3-4 and shootouts 5 where played",
                     "value": str(periods), "passed": None})
        rows.append({"check": f"{t}: shootout matches", "target": "listed",
                     "value": str(int(m["shootout"].sum())), "passed": None})
        xi = ev[ev["type"] == "Starting XI"].groupby("match_id").size()
        ok = int((xi == 2).sum())
        rows.append({"check": f"{t}: matches with two Starting XI events", "target": f"{n} of {n}",
                     "value": f"{ok} of {n}", "passed": ok == n})
        hosted = int((~m["neutral_ground"]).sum())
        rows.append({"check": f"{t}: matches with a host team (neutral_ground false)", "target": "listed",
                     "value": str(hosted), "passed": None})
    rows.append({"check": "all: empty in open data (no values)", "target": "empty",
                 "value": "obv_total_net, duel/pass/shot/clearance/miscontrol_duel_win_probability, "
                          "location_bucket, "
                          "possession_directness, control_degree, is_transition, is_controlled_possession",
                 "passed": None})
    return pd.DataFrame(rows, columns=["check", "target", "value", "passed"]).astype({"passed": "boolean"})


def load_tournament(cfg: Config, tournament: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(events of every period, matches) for one senior tournament label."""
    competition_id, season_id = COMPETITIONS[tournament]
    matches_json, ids = fetch(cfg, competition_id, season_id)
    base, root = cfg.external_urls["statsbomb_open_data"], open_data_dir(cfg)
    frames = [events_frame(mid, read_json(base, f"events/{mid}.json", root)) for mid in ids]
    events = prepare_senior_events(pd.concat(frames, ignore_index=True), tournament)
    return events, raw_matches(matches_json)


def run(cfg: Config) -> None:
    kept_events, all_events, matches = [], [], []
    for t in COMPETITIONS:
        ev, raw = load_tournament(cfg, t)
        kept, shootouts = sb.split_shootouts(ev)
        all_events.append(ev[["tournament", "match_id", "period", "type", "team_id", "shot_outcome"]])
        kept_events.append(kept)
        m = sb.prepare_matches(raw, t, shootouts)
        matches.append(m)
        n_shootouts = int(m["shootout"].sum())
        print(f"  {t}: {len(m)} matches, {len(ev):,} events, {n_shootouts} shootouts", flush=True)

    matches_df = pd.concat(matches, ignore_index=True)
    checks = check_open_data(pd.concat(all_events, ignore_index=True), matches_df)
    write_table(cfg, "senior_events", pd.concat(kept_events, ignore_index=True))
    write_table(cfg, "senior_matches", matches_df)
    write_table(cfg, "checks_open_data", checks)
    with pd.option_context("display.width", 200, "display.max_colwidth", 120):
        print(checks.to_string(index=False))
