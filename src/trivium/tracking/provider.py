"""Provider check (test 11): StatsBomb and FIFA figures side by side for the same team-match (men).

The providers define events differently, so the two values are labelled with their own
definitions and never mixed in one measure. Distance compares FIFA's physical file with our
broadcast-tracking totals (the statsbomb_value column holds the tracking value for that row,
as its definition says). Periods 1-2 only (no shootout kicks).

Output table: provider_check_team_match (written by the tracking stage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trivium.teams import clean_team_name

KEYS = ["tournament", "match_id", "team"]
SB_ON_TARGET = ("Goal", "Saved", "Saved to Post")

# measure -> (StatsBomb column, FIFA column, StatsBomb definition, FIFA definition)
MEASURES = {
    "possession_pct": (
        "sb_possession_pct", "possession_pct",
        "StatsBomb: share of both teams' possession time, each possession timed from its first to its last event",
        "FIFA report: share of match time in possession; a separate in-contest share is not given to either team",
    ),
    "shots": ("sb_shots", "attempts", "StatsBomb: Shot events (penalties included, shootouts excluded)",
              "FIFA report: attempts at goal"),
    "shots_on_target": ("sb_shots_on_target", "attempts_on_target",
                        "StatsBomb: shots with outcome Goal, Saved or Saved to Post", "FIFA report: attempts on target"),
    "passes": ("sb_passes", "passes", "StatsBomb: Pass events", "FIFA report: total passes"),
    "passes_complete": ("sb_passes_complete", "passes_complete", "StatsBomb: passes with no outcome (completed)",
                        "FIFA report: completed passes"),
    "pass_completion_pct": ("sb_pass_completion_pct", "pass_completion_pct",
                            "StatsBomb: completed passes / passes", "FIFA report: pass completion %"),
    "pressures": ("sb_pressures", "pressures_applied", "StatsBomb: Pressure events",
                  "FIFA report: defensive pressures applied"),
    "distance_km": ("distance_km_tracking", "distance_km",
                    "Broadcast tracking (this project): sum of speed / 25 over every frame, all players",
                    "FIFA physical file: team total distance"),
}
COLUMNS = [*KEYS, "measure", "statsbomb_value", "fifa_value", "statsbomb_definition", "fifa_definition"]


def statsbomb_team_match(events: pd.DataFrame) -> pd.DataFrame:
    """StatsBomb figures per team-match from the events table (periods 1-2)."""
    ev = events[events["period"].isin([1, 2])]
    by_team = ev.groupby(KEYS)
    out = pd.DataFrame({
        "sb_shots": by_team["type"].apply(lambda t: (t == "Shot").sum()),
        "sb_shots_on_target": ev[(ev["type"] == "Shot") & ev["shot_outcome"].isin(SB_ON_TARGET)].groupby(KEYS).size(),
        "sb_passes": by_team["type"].apply(lambda t: (t == "Pass").sum()),
        "sb_passes_complete": ev[(ev["type"] == "Pass") & ev["pass_outcome"].isna()].groupby(KEYS).size(),
        "sb_pressures": by_team["type"].apply(lambda t: (t == "Pressure").sum()),
    }).fillna(0.0)
    out["sb_pass_completion_pct"] = 100 * out["sb_passes_complete"] / out["sb_passes"].where(out["sb_passes"] > 0)

    # Possession time: each possession from its first to its last event, credited to the possession team.
    poss = ev.assign(team=ev["possession_team"].map(clean_team_name)).groupby(
        ["tournament", "match_id", "period", "possession", "team"])["period_seconds"].agg(["min", "max"])
    seconds = (poss["max"] - poss["min"]).groupby(level=KEYS).sum()
    share = seconds / seconds.groupby(level=["tournament", "match_id"]).transform("sum")
    out["sb_possession_pct"] = 100 * share
    return out.reset_index()


def provider_check(statsbomb: pd.DataFrame, report: pd.DataFrame, provider_table: pd.DataFrame | None) -> pd.DataFrame:
    """Long table: one row per team-match x measure with both providers' values and definitions."""
    wide = statsbomb.merge(report, on=KEYS, how="inner", validate="one_to_one")
    if provider_table is not None:
        wide = wide.drop(columns=["distance_km"], errors="ignore").merge(
            provider_table[[*KEYS, "distance_km", "distance_km_tracking"]], on=KEYS, how="left", validate="one_to_one")
    rows = []
    for measure, (sb_col, fifa_col, sb_def, fifa_def) in MEASURES.items():
        if sb_col not in wide or fifa_col not in wide:
            continue
        part = wide[KEYS].assign(measure=measure, statsbomb_value=wide[sb_col].astype(float),
                                 fifa_value=wide[fifa_col].astype(float), statsbomb_definition=sb_def,
                                 fifa_definition=fifa_def)
        rows.append(part)
    return pd.concat(rows, ignore_index=True)[COLUMNS] if rows else pd.DataFrame(columns=COLUMNS)


def provider_summary(check: pd.DataFrame) -> pd.DataFrame:
    """Per measure: team-matches, correlation and median ratio StatsBomb / FIFA (information, not pass/fail)."""
    def one(g: pd.DataFrame) -> pd.Series:
        both = g.dropna(subset=["statsbomb_value", "fifa_value"])
        ratio = (both["statsbomb_value"] / both["fifa_value"].where(both["fifa_value"] != 0)).median()
        corr = both["statsbomb_value"].corr(both["fifa_value"]) if len(both) > 2 else np.nan
        return pd.Series({"team_matches": len(both), "correlation": corr, "median_ratio": ratio})
    return check.groupby("measure", sort=False).apply(one).reset_index()
