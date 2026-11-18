"""Average outfield positions per 5-minute window and possession state (men), for formations.

One second in every 25 frames of each match's tracking file is read (the other lines are not
parsed), straight from the zip. Each outfield player's position is turned into the team's
attacking direction (x from 0 at the team's own goal line to 105, y flipped with x so the team's
left stays on one side), and averaged per team, period, 5-minute window of the period clock,
possession state (with or without the ball; out of play left out) and player. Goalkeepers,
extra time and matches left out of the tracking measures (more than 30% of positions filled
in by the provider) are not used. No names are read into the table; person_id is internal.

Output tables: tracking/<match_id>/positions_window (per match, skipped when it exists),
positions_window, formation_team_window and formation_team_match.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from trivium.config import Config
from trivium.storage import read_table, table_exists, write_table
from trivium.tracking.running import attacking_x, possession_state, position_group
from trivium.tracking.formations import team_match_formations, team_window_formations
from trivium.tracking.stream import PERIODS, TRACKING_MEMBER, _loads, _member, parse_clock, read_metadata

PITCH_WIDTH = 68.0
EVERY = 25  # frames: one position a second
WINDOW_S = 300  # 5-minute windows of the period clock
STATES = ("with_ball", "without_ball")
COLUMNS = ["tournament", "match_id", "team", "period", "window", "window_start_min", "state", "person_id",
           "position_group", "x_m", "y_m", "seconds"]
WORKERS = 3


def sampled_frames(zip_path: str | Path, every: int = EVERY) -> Iterator[dict[str, Any]]:
    """Every `every`-th frame of the tracking file; the lines in between are skipped unparsed."""
    with zipfile.ZipFile(zip_path) as zf, zf.open(_member(zf, TRACKING_MEMBER)) as lines:
        for n, line in enumerate(lines):
            if n % every == 0 and line.strip():
                yield _loads(line)


def match_positions(zip_path: str | Path, every: int = EVERY) -> pd.DataFrame:
    """One match: tracking team name, period, window, state, person_id, position_group and mean x, y
    (attacking direction) with the seconds averaged."""
    meta = read_metadata(zip_path)
    groups = {p["ID"]: position_group(p.get("Position")) for side in ("HomeTeam", "AwayTeam")
              for p in meta[side]["Players"]}
    rows = []
    for frame in sampled_frames(zip_path, every):
        period = PERIODS.get(frame["period"])
        if period not in (1, 2):
            continue
        side, possession = frame["pitch_side"], frame["ball_possession"]
        clock = parse_clock(frame["time"])
        for p in frame["players"]:
            if p.get("category_name") != "player" or p.get("x") is None or p.get("y") is None:
                continue
            team = p["team_name"]
            state = possession_state(possession, team, side)
            if state not in STATES:
                continue
            flipped = attacking_x(0.0, team, side) != 0.0  # the team attacks right to left in this frame
            x = attacking_x(p["x"], team, side)
            y = PITCH_WIDTH - p["y"] if flipped else p["y"]
            rows.append((team, period, int(clock // WINDOW_S), state, int(p["person_id"]), x, y))
    df = pd.DataFrame(rows, columns=["team", "period", "window", "state", "person_id", "x_m", "y_m"])
    out = df.groupby(["team", "period", "window", "state", "person_id"], as_index=False).agg(
        x_m=("x_m", "mean"), y_m=("y_m", "mean"), seconds=("x_m", "size"))
    out["position_group"] = out["person_id"].map(groups).fillna("Unknown")
    out["window_start_min"] = (out["period"] - 1) * 45 + out["window"] * WINDOW_S / 60
    return out


def _one(args: tuple[Config, int, str]) -> int:
    cfg, match_id, zip_path = args
    name = f"tracking/{match_id}/positions_window"
    if not table_exists(cfg, name):
        write_table(cfg, name, match_positions(zip_path))
    return match_id


def run(cfg: Config) -> None:
    from trivium.ingest.match_folders import match_folders, men_matches
    from trivium.tracking.running import excluded_matches
    from trivium.tracking.stream import with_statsbomb_teams

    folders = match_folders(cfg)
    ready = folders[folders["matched"] & folders["tracking_zip"].notna()]
    running = read_table(cfg, "running_player_match") if table_exists(cfg, "running_player_match") else None
    left_out = set(excluded_matches(running)["match_id"].astype(int))
    todo = [(cfg, int(f.match_id), str(f.tracking_zip)) for f in ready.itertuples()
            if int(f.match_id) not in left_out]
    with ProcessPoolExecutor(WORKERS) as pool:
        for done, match_id in enumerate(pool.map(_one, todo), 1):
            print(f"[positions] {done} of {len(todo)} (match {match_id})", flush=True)
    parts = [read_table(cfg, f"tracking/{m}/positions_window").assign(match_id=m) for _, m, _ in todo]
    table = with_statsbomb_teams(pd.concat(parts, ignore_index=True), men_matches(cfg))
    table["team"] = table["team"].fillna(table["tracking_team"])
    table.insert(0, "tournament", "men")
    write_table(cfg, "positions_window", table[COLUMNS])
    print(f"positions_window: {len(table):,} rows, {table['match_id'].nunique()} matches, "
          f"{len(left_out)} matches left out (positions filled in)")
    window_formations = team_window_formations(table[COLUMNS])
    write_table(cfg, "formation_team_window", window_formations)
    match_formations = team_match_formations(window_formations)
    write_table(cfg, "formation_team_match", match_formations)
    print(f"formation_team_window: {len(window_formations):,} windows; "
          f"formation_team_match: {len(match_formations):,} team-matches")
