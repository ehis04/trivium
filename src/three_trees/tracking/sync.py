"""Tracking-event sync and coordinates.

Time offset per period per match (tracking time = StatsBomb time + offset)
by grid search over passes and shots: first every whole second from -120 s to
+120 s, then 0.2 s steps within a second of the best (some halves are offset by
about 30 s, outside the planned -15 to +15 s). Tracking to StatsBomb coordinates:
mirror when the event team is not pitch_side.left (x = 105 - x, y = 68 - y),
then x_sb = x * 120 / 105, y_sb = y * 80 / 68.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from three_trees.ingest.match_folders import team_key

# Pitch sizes: tracking in metres (origin top-left), StatsBomb in yards.
TRACKING_LENGTH, TRACKING_WIDTH = 105.0, 68.0
SB_LENGTH, SB_WIDTH = 120.0, 80.0

# Candidate offsets: every whole second up to SEARCH_LIMIT_S either way, then FINE_STEP_S steps
# within a second of the best.
SEARCH_LIMIT_S = 120.0
COARSE_OFFSETS = np.arange(-SEARCH_LIMIT_S, SEARCH_LIMIT_S + 1e-9, 1.0)
FINE_STEP_S = 0.2

# StatsBomb event types used for the sync: the ball is at the event location at the event time.
SYNC_TYPES = ("Pass", "Shot")

# Columns the sync needs from the StatsBomb events table.
EVENT_COLUMNS = ["match_id", "period", "period_seconds", "type", "team", "location_x", "location_y"]
SYNC_COLUMNS = ["match_id", "period", "offset_s", "median_error_yd", "n_events"]


def left_teams(meta: dict[str, Any]) -> dict[int, str]:
    """Per period, the tracking team name attacking left to right (pitch_side.left).

    From metadata.json: Phase<n>HomeGKLeft true means the home team is pitch_side.left
    in that period. Periods that did not happen (start frame 0) are left out.
    """
    home, away = meta["HomeTeam"]["Name"], meta["AwayTeam"]["Name"]
    return {
        n: home if meta[f"Phase{n}HomeGKLeft"] else away
        for n in range(1, 5)
        if meta.get(f"Phase{n}StartFrame")
    }


def to_statsbomb(x: np.ndarray, y: np.ndarray, mirror: np.ndarray | bool) -> tuple[np.ndarray, np.ndarray]:
    """Tracking metres -> StatsBomb yards, from the event team's view.

    mirror is True where the event team is not pitch_side.left (it attacks right to left
    in the tracking), so its view is turned round: x = 105 - x, y = 68 - y.
    """
    x = np.where(mirror, TRACKING_LENGTH - x, x)
    y = np.where(mirror, TRACKING_WIDTH - y, y)
    return x * SB_LENGTH / TRACKING_LENGTH, y * SB_WIDTH / TRACKING_WIDTH


def nearest_frames(ball_t: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Index of the tracking frame nearest each time in t (ball_t sorted ascending)."""
    right = np.clip(np.searchsorted(ball_t, t), 1, len(ball_t) - 1)
    left = right - 1
    return np.where(np.abs(t - ball_t[left]) <= np.abs(ball_t[right] - t), left, right)


def best_offset(
    event_t: np.ndarray, event_x: np.ndarray, event_y: np.ndarray, mirror: np.ndarray,
    ball_t: np.ndarray, ball_x: np.ndarray, ball_y: np.ndarray,
) -> tuple[float, float, int]:
    """Grid search for one period: (offset_s, median_error_yd, n_events).

    For each candidate offset, every event is paired with the tracking frame nearest
    event time + offset; the error is the distance in yards between the ball there
    (turned to the event team's view) and the event location. Lowest median error wins:
    first over whole seconds (COARSE_OFFSETS), then in FINE_STEP_S steps around the best.
    """
    order = np.argsort(ball_t)
    ball_t, ball_x, ball_y = ball_t[order], ball_x[order], ball_y[order]

    def errors(offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # One row per candidate offset, one column per event.
        idx = nearest_frames(ball_t, event_t[None, :] + offsets[:, None])
        bx, by = to_statsbomb(ball_x[idx], ball_y[idx], mirror[None, :])
        error = np.hypot(bx - event_x[None, :], by - event_y[None, :])
        return error, np.nanmedian(error, axis=1)  # frames with no ball position are left out

    _, coarse = errors(COARSE_OFFSETS)
    centre = COARSE_OFFSETS[int(np.nanargmin(coarse))]
    fine_offsets = np.round(np.arange(centre - 1.0, centre + 1.0 + 1e-9, FINE_STEP_S), 1)
    error, median = errors(fine_offsets)

    best = int(np.nanargmin(median))
    n = int(np.isfinite(error[best]).sum())
    return float(fine_offsets[best]), float(median[best]), n


def sync_match(events: pd.DataFrame, ball: pd.DataFrame, left: dict[int, str]) -> pd.DataFrame:
    """Sync offsets for one match: one row per period (match_id, period, offset_s, median_error_yd, n_events).

    events: the match's StatsBomb events (EVENT_COLUMNS); only passes and shots with a location are used.
    ball: the match's ball table (period, clock_s, ball_x, ball_y).
    left: per period, the tracking team attacking left to right (left_teams).
    """
    ev = events[events["type"].isin(SYNC_TYPES) & events["location_x"].notna()]
    rows = []
    for period, left_team in sorted(left.items()):
        e = ev[ev["period"] == period]
        b = ball[ball["period"] == period]
        if e.empty or b.empty:
            continue
        # The event team is mirrored when it is not the team attacking left to right.
        mirror = (team_key(e["team"]) != team_key(pd.Series([left_team])).iloc[0]).to_numpy()
        offset, error, n = best_offset(
            e["period_seconds"].to_numpy(float), e["location_x"].to_numpy(float), e["location_y"].to_numpy(float),
            mirror, b["clock_s"].to_numpy(float), b["ball_x"].to_numpy(float), b["ball_y"].to_numpy(float),
        )
        rows.append({"match_id": int(ev["match_id"].iloc[0]), "period": period, "offset_s": offset,
                     "median_error_yd": error, "n_events": n})
    return pd.DataFrame(rows, columns=SYNC_COLUMNS)
