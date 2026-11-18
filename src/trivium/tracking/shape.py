"""Team shape at 1 Hz: defensive line height, team length and width.

Taken from every 25th tracking frame (once a second), outfield players only (goalkeepers left
out), with x turned to the team's attacking direction so x = 0 is the team's own goal line:
  line_height_m   mean x of the four deepest outfield players (metres from own goal)
  team_length_m   deepest to highest outfield player (metres)
  team_width_m    widest outfield spread across the pitch (metres)
A second is left out for a team with fewer than four outfield players located.
Limits: broadcast tracking fills in players the camera cannot see (is_imputed); those
positions are used as given.

Output table: shape_team_second (written by the tracking stage).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

SAMPLE_EVERY = 25  # frames: one shape per second at 25 frames a second
DEEPEST = 4        # outfield players averaged for the defensive line
SHAPE_COLUMNS = ["line_height_m", "team_length_m", "team_width_m"]


def team_shape(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """(line height, length, width) of one team in one frame, from its outfield players' positions.

    x is in the team's attacking direction (0 = own goal line). Fewer than DEEPEST players: all NaN.
    """
    if len(x) < DEEPEST:
        return math.nan, math.nan, math.nan
    xs = np.sort(np.asarray(x, dtype=float))
    ys = np.asarray(y, dtype=float)
    return float(xs[:DEEPEST].mean()), float(xs[-1] - xs[0]), float(ys.max() - ys.min())


def shape_means(shape: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Match means of the shape columns over the seconds the team was without the ball."""
    without = shape[shape["state"] == "without_ball"]
    return without.groupby(keys, as_index=False)[SHAPE_COLUMNS].mean()
