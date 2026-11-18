"""Team running in the 10 seconds before each StatsBomb defensive action (men).

For each Duel (Tackle), Interception, Pressure and Ball Recovery in periods 1-2, the event's
time in the tracking clock is its StatsBomb time + that half's sync offset. The window is the
10 whole seconds of the clock before the event's second; team_hsr_10s_m and
team_distance_10s_m are the acting team's totals over all its players in those seconds.
Limits: seconds are whole clock seconds, so the window ends up to 1 s before the event; events
in a half with no sync offset are left out.

Output table: event_running_windows (written by the tracking stage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_S = 10
ACTION_TYPES = ("Interception", "Pressure", "Ball Recovery")  # plus Duel with duel_type Tackle
EVENT_COLUMNS = ["id", "match_id", "period", "period_seconds", "type", "duel_type", "team", "location_x",
                 "location_y"]
WINDOW_COLUMNS = ["tournament", "match_id", "event_id", "team", "type", "location_x", "location_y",
                  "team_hsr_10s_m", "team_distance_10s_m"]


def defensive_actions(events: pd.DataFrame) -> pd.DataFrame:
    """The StatsBomb events the windows are for: tackles, interceptions, pressures, ball recoveries (periods 1-2)."""
    tackle = (events["type"] == "Duel") & (events["duel_type"] == "Tackle")
    keep = (tackle | events["type"].isin(ACTION_TYPES)) & events["period"].isin([1, 2])
    return events[keep]


def event_windows(actions: pd.DataFrame, seconds: pd.DataFrame, offsets: dict[int, float]) -> pd.DataFrame:
    """Windows for one match.

    actions: that match's defensive actions (team = StatsBomb name). seconds: the match's per-second
    team table with team = StatsBomb name. offsets: period -> sync offset (tracking = StatsBomb + offset).
    """
    team_seconds = seconds.groupby(["team", "period", "second"])[["hsr_m", "distance_m"]].sum()
    rows = []
    for (team, period), acts in actions.groupby(["team", "period"]):
        if period not in offsets or (team, period) not in team_seconds.index.droplevel("second"):
            continue
        s = team_seconds.loc[(team, period)]
        last = int(s.index.max()) + 1
        # Running totals up to each second, so each window is one subtraction.
        dense = s.reindex(range(last), fill_value=0.0)
        cum = {c: np.concatenate([[0.0], np.cumsum(dense[c].to_numpy())]) for c in ("hsr_m", "distance_m")}
        end = np.clip(np.floor(acts["period_seconds"].to_numpy() + offsets[period]).astype(int), 0, last)
        start = np.clip(end - WINDOW_S, 0, last)
        rows.append(pd.DataFrame({
            "event_id": acts["id"].to_numpy(), "team": team, "type": acts["type"].to_numpy(),
            "location_x": acts["location_x"].to_numpy(), "location_y": acts["location_y"].to_numpy(),
            "team_hsr_10s_m": cum["hsr_m"][end] - cum["hsr_m"][start],
            "team_distance_10s_m": cum["distance_m"][end] - cum["distance_m"][start],
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=WINDOW_COLUMNS[2:])
