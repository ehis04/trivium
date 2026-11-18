"""Out-of-play share: broadcast tracking's ball_possession against FIFA's ball_alive (men).

Our tracking marks a frame "out of play" in ball_possession; FIFA's own tracking file
(tracking_data_fifa_ifes.jsonl, in the same zip) marks each frame ball_alive true or false.
Frames are lined up by period and frame number within the period (FIFA frame_count, our clock x 25; the
same 25 fps grid). The comparison uses the seconds of the team shape table (every 25th frame),
where our possession state is already saved: out_of_play there = ball_possession not a team.
Per match: each provider's out-of-play share over the seconds both have, and how often they agree.
Limits: FIFA's file starts a few frames later in each period; seconds only one file has are left out.

Output: tracking/<match_id>/ball_alive per match; shares in checks_tracking.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from three_trees.tracking.stream import iter_frames

FIFA_MEMBER = "tracking_data_fifa_ifes.jsonl"
FPS = 25


def frame_key(clock_s: pd.Series | np.ndarray) -> np.ndarray:
    """Clock seconds -> whole frame number within the period (25 per second), for lining up two files."""
    return np.rint(np.asarray(clock_s, dtype=float) * FPS).astype(np.int64)


def fifa_ball_alive(zip_path: str | Path) -> pd.DataFrame:
    """period, frame (within the period) and ball_alive for every frame of FIFA's tracking file.

    FIFA's timestamp runs on from the first half into the second, but its frame_count restarts each
    period at the same 25 fps clock as ours (frame_count / 25 = our clock), so frame_count is the key.
    """
    try:
        rows = [(int(f["match_period"]), int(f["frame_count"]), bool(f["ball_alive"]))
                for f in iter_frames(zip_path, FIFA_MEMBER) if f.get("ball_alive") is not None]
    except KeyError:  # no FIFA tracking file in this zip
        rows = []
    df = pd.DataFrame(rows, columns=["period", "frame", "ball_alive"])
    return df.drop_duplicates(["period", "frame"])


def out_of_play_shares(shape: pd.DataFrame, alive: pd.DataFrame) -> dict[str, float]:
    """Out-of-play shares over the seconds both files have.

    shape: the match's shape table (period, clock_s, state per team). alive: fifa_ball_alive.
    """
    ours = shape.drop_duplicates(["period", "clock_s"]).assign(frame=lambda d: frame_key(d["clock_s"]))
    ours = ours.assign(tracking_out=ours["state"] == "out_of_play")[["period", "frame", "tracking_out"]]
    both = ours.merge(alive, on=["period", "frame"], how="inner", validate="one_to_one")
    fifa_out = ~both["ball_alive"]
    return {
        "seconds": float(len(both)),
        "tracking_out_of_play_share": float(both["tracking_out"].mean()) if len(both) else np.nan,
        "fifa_out_of_play_share": float(fifa_out.mean()) if len(both) else np.nan,
        "agreement_share": float((both["tracking_out"] == fifa_out).mean()) if len(both) else np.nan,
    }
