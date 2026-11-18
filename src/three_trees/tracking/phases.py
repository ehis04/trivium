"""FIFA phases of play from each match's Enhanced file, joined to the tracking.

The Enhanced file holds video clips in rows named "<TEAM>_<PHASE>". Clip times are video
seconds; the KICK OFF row has one clip per half ("First Half", "Second Half") whose start is
that half's kick-off, so clock seconds in a half = video seconds - that half's kick-off.
Rows for single players ("<TEAM>_<shirt> <name>") are never read.

Each whole second of the tracking clock (its midpoint) takes every phase clip of the team
covering it. Phase clips overlap (about 31% of clip time is covered by two phases of the same
team), so a second can count in more than one phase and phase totals add to more than the
match total. fifa_<phase>_share = the team's clip time in that phase / its clip time in all
phases. FIFA codes BLOCK as an eleventh phase; it is kept as "block".

Output tables: phase_clips, running_fifa_phase; fifa_<phase>_share columns of team_match_provider.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import orjson
import pandas as pd

# FIFA row names -> phase names in our tables.
PHASES = {
    "BUILD UP": "build_up",
    "PROGRESSION": "progression",
    "FINAL THIRD": "final_third",
    "LONG BALL": "long_ball",
    "ATTACKING TRANSITION": "attacking_transition",
    "COUNTER ATTACK": "counter_attack",
    "PRESS": "press",
    "COUNTER PRESS": "counter_press",
    "DEFENSIVE TRANSITION": "defensive_transition",
    "RECOVERY": "recovery",
    "BLOCK": "block",
}
HALVES = {"First Half": 1, "Second Half": 2, "Extra Time First Half": 3, "Extra Time Second Half": 4}
PLAYER_ROW = re.compile(r"^[^_]+_\d+\s")  # "<TEAM>_<shirt> <name>": skipped, names never read
CLIP_COLUMNS = ["fifa_team", "phase", "period", "start_s", "end_s"]
RUNNING_COLUMNS = ["seconds", "distance_m", "hsr_m", "sprint_m"]


def kick_offs(rows: list[dict]) -> dict[int, float]:
    """Period number -> video seconds of that half's kick-off, from the KICK OFF row."""
    out = {}
    for row in rows:
        if row["row_name"] != "KICK OFF":
            continue
        for clip in row["clips"]:
            labels = [q.get("name") for q in clip.get("qualifiers", {}).get("qualifiers_array", [])
                      if q.get("category") == "GAME TIME"]
            period = next((HALVES[label] for label in labels if label in HALVES), None)
            if period is not None:
                out[period] = float(clip["time_start"])
    return out


def to_clock(video_s: float, kicks: dict[int, float]) -> tuple[int | None, float]:
    """(period, clock seconds in that period) for a video time: the latest kick-off at or before it."""
    started = [(p, t) for p, t in kicks.items() if t <= video_s]
    if not started:
        return None, float("nan")
    period, kick = max(started, key=lambda pt: pt[1])
    return period, video_s - kick


def read_phase_clips(path: str | Path) -> pd.DataFrame:
    """Phase clips of both teams in one Enhanced file: fifa_team, phase, period, start_s, end_s (clock).

    Clips that start before the first kick-off are left out.
    """
    rows = [r for r in orjson.loads(Path(path).read_bytes())["rows"] if not PLAYER_ROW.match(r["row_name"])]
    kicks = kick_offs(rows)
    out = []
    for row in rows:
        team, _, name = row["row_name"].partition("_")
        if name not in PHASES:
            continue
        for clip in row["clips"]:
            period, start = to_clock(float(clip["time_start"]), kicks)
            if period is None:
                continue
            out.append((team, PHASES[name], period, start, float(clip["time_end"]) - kicks[period]))
    return pd.DataFrame(out, columns=CLIP_COLUMNS)


def clip_seconds(clips: pd.DataFrame) -> pd.DataFrame:
    """One row per team x phase x period x whole second whose midpoint lies inside one of the clips."""
    parts = []
    for (team, phase, period), c in clips.groupby(["team", "phase", "period"]):
        secs = [np.arange(np.ceil(s - 0.5), np.ceil(e - 0.5)) for s, e in zip(c["start_s"], c["end_s"])]
        secs = np.unique(np.concatenate(secs)) if secs else np.array([])
        parts.append(pd.DataFrame({"team": team, "phase": phase, "period": period, "second": secs.astype(int)}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["team", "phase", "period", "second"])


def running_by_phase(clips: pd.DataFrame, seconds: pd.DataFrame) -> pd.DataFrame:
    """Team x position group x phase: seconds in the phase and the running in those seconds.

    clips: phase clips with a team column (the same names as seconds). seconds: the match's
    per-second team table from running.match_pass.
    """
    joined = clip_seconds(clips).merge(seconds, on=["team", "period", "second"], how="inner")
    out = joined.groupby(["team", "position_group", "phase"], as_index=False).agg(
        seconds=("second", "size"), distance_m=("distance_m", "sum"), hsr_m=("hsr_m", "sum"),
        sprint_m=("sprint_m", "sum"))
    return out


def phase_shares(clips: pd.DataFrame) -> pd.DataFrame:
    """One row per team: fifa_<phase>_share = clip time in the phase / clip time in all phases."""
    time = clips.assign(duration=clips["end_s"] - clips["start_s"]).pivot_table(
        index="team", columns="phase", values="duration", aggfunc="sum", fill_value=0.0)
    time = time.reindex(columns=list(PHASES.values()), fill_value=0.0)
    shares = time.div(time.sum(axis=1), axis=0)
    shares.columns = [f"fifa_{p}_share" for p in shares.columns]
    return shares.reset_index()
