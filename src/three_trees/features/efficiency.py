"""Efficiency index (supplement addition 3): one score per team-match and per
team-tournament combining the cost of dominance and conversion.

- Cost: men high-speed running without the ball per point of dominance (tracking);
  women, or men without tracking, defensive work (pressures and defensive actions)
  per point of dominance. Lower cost is better, so its z-score is negated.
- Conversion: non-penalty xG per point of dominance. Over-performance is not used,
  so the index can be tested against over-performance without circularity.
- z-scores within tournament (team-match rows for the team-match table, teams for
  the team-tournament table); efficiency_index = mean of the two z-scores. Every z
  is oriented so that higher is more efficient.
- Position groups (men): high-speed running without the ball per match by position
  group, per point of the team's dominance, z within tournament and position group.

Output tables: efficiency_team_match, efficiency_team_tournament,
efficiency_position_group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from three_trees.measures import measure

COST_MEASURES = {
    "men": ("hsr_without_ball_per_dominance", "defensive_work_per_dominance"),
    "women": ("defensive_work_per_dominance",),
}
KEYS = ["tournament", "match_id", "team"]


def _col(name: str) -> str:
    return str(measure(name)["column"])


def _z(values: pd.Series) -> pd.Series:
    sd = values.std()
    return (values - values.mean()) / sd if sd and np.isfinite(sd) else values * 0.0


def cost_measure(team_match: pd.DataFrame, tournament: str) -> str | None:
    """The first cost measure with data for the tournament, or None."""
    rows = team_match[team_match["tournament"] == tournament]
    for name in COST_MEASURES[tournament]:
        col = _col(name)
        if col in rows and rows[col].notna().any():
            return name
    return None


def _values(team_match: pd.DataFrame, tournament: str) -> pd.DataFrame:
    rows = team_match[team_match["tournament"] == tournament].copy()
    name = cost_measure(team_match, tournament)
    rows["cost_measure"] = name
    rows["cost_value"] = rows[_col(name)] if name else np.nan
    rows["conversion_value"] = rows[_col("np_xg")] / (rows[_col("dominance")] * 100)
    return rows


def efficiency_team_match(team_match: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for tournament in team_match["tournament"].drop_duplicates():
        rows = _values(team_match, tournament)
        rows["cost_z"] = -_z(rows["cost_value"])
        rows["conversion_z"] = _z(rows["conversion_value"])
        rows["efficiency_index"] = rows[["cost_z", "conversion_z"]].mean(axis=1, skipna=False)
        parts.append(rows[[*KEYS, "cost_measure", "cost_value", "conversion_value", "cost_z", "conversion_z",
                           "efficiency_index"]])
    return pd.concat(parts, ignore_index=True)


def efficiency_team_tournament(team_match: pd.DataFrame) -> pd.DataFrame:
    """Team means over matches, standardised across the tournament's teams, then ranked."""
    parts = []
    for tournament in team_match["tournament"].drop_duplicates():
        rows = _values(team_match, tournament)
        teams = rows.groupby(["tournament", "team"], as_index=False).agg(
            matches=("match_id", "nunique"),
            cost_measure=("cost_measure", "first"),
            cost_value=("cost_value", "mean"),
            conversion_value=("conversion_value", "mean"),
        )
        teams["cost_z"] = -_z(teams["cost_value"])
        teams["conversion_z"] = _z(teams["conversion_value"])
        teams["efficiency_index"] = teams[["cost_z", "conversion_z"]].mean(axis=1, skipna=False)
        teams["efficiency_rank"] = teams["efficiency_index"].rank(ascending=False, method="min").astype("Int64")
        parts.append(teams.sort_values("efficiency_rank", ignore_index=True))
    return pd.concat(parts, ignore_index=True)


def efficiency_position_group(running: pd.DataFrame | None, team_match: pd.DataFrame) -> pd.DataFrame:
    """Men: high-speed running without the ball by position group per point of dominance."""
    columns = ["tournament", "team", "position_group", "matches", "hsr_without_ball_m_per_match",
               "cost_per_dominance", "cost_z"]
    if running is None or running.empty or "position_group" not in running:
        return pd.DataFrame(columns=columns)
    without = running[running["state"] == "without_ball"]
    per_match = without.groupby(["tournament", "match_id", "team", "position_group"], as_index=False)["hsr_m"].sum()
    per_match = per_match.merge(team_match[[*KEYS, _col("dominance")]], on=KEYS, how="inner")
    per_match["cost_per_dominance"] = per_match["hsr_m"] / (per_match[_col("dominance")] * 100)
    groups = per_match.groupby(["tournament", "team", "position_group"], as_index=False).agg(
        matches=("match_id", "nunique"),
        hsr_without_ball_m_per_match=("hsr_m", "mean"),
        cost_per_dominance=("cost_per_dominance", "mean"),
    )
    groups["cost_z"] = -groups.groupby(["tournament", "position_group"])["cost_per_dominance"].transform(_z)
    return groups[columns].sort_values(["tournament", "position_group", "cost_z"], ascending=[True, True, False],
                                       ignore_index=True)
