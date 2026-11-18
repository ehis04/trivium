"""Load spread and threat spread per team-match.

Load spread: minutes_outside_top11_share (share of the team's minutes in the match
played by players outside its 11 players with most tournament minutes; ties broken by
player id) and starters_changed (starters not in the previous match's starting XI;
NaN for match 1).

Threat spread: each player's non-penalty xG plus xA in the match (internal, never
written); top2_xg_xa_share (the two highest players' xG + xA over the team's xG + xA,
NaN if 0); chance_zone_spread (normalised entropy, 0-1, of the team's xG by shot zone
plus xA by assisting pass zone over the 18 zones of features.dominance.zone_of; NaN if
0). xA is the non-penalty xG of the shot a pass assisted (features.resources).

Limits: player values are rolled up to the team; top3_hsr_share (men) comes from the
tracking stage.

Output: columns merged into team_match by features.tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from three_trees.features import dominance as dom
from three_trees.features import resources as res

TEAM_KEYS = res.TEAM_KEYS
TOP_PLAYERS = 11


def minutes_outside_top11(minutes: pd.DataFrame) -> pd.DataFrame:
    """Per team-match: minutes_outside_top11_share."""
    totals = (minutes.groupby(["tournament", "team_id", "player_id"], as_index=False)["minutes"].sum()
              .sort_values(["tournament", "team_id", "minutes", "player_id"],
                           ascending=[True, True, False, True]))
    top = totals.groupby(["tournament", "team_id"]).head(TOP_PLAYERS)[["tournament", "team_id", "player_id"]]
    pm = minutes.merge(top.assign(top11=True), on=["tournament", "team_id", "player_id"], how="left")
    pm["outside"] = pm["minutes"].where(pm["top11"].isna(), 0.0)
    out = pm.groupby(TEAM_KEYS).agg(team_minutes=("minutes", "sum"), outside=("outside", "sum"))
    out["minutes_outside_top11_share"] = out["outside"] / out["team_minutes"].where(out["team_minutes"] > 0)
    return out[["minutes_outside_top11_share"]].reset_index()


def starters_changed(minutes: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Per team-match: starters not in the team's previous starting XI (NaN for match 1)."""
    starters = minutes[minutes["started"]].merge(
        context[TEAM_KEYS + ["match_number"]], on=TEAM_KEYS, how="left", validate="many_to_one")
    previous = starters[["tournament", "team_id", "player_id", "match_number"]].assign(
        match_number=lambda d: d["match_number"] + 1, kept=True)
    joined = starters.merge(previous, on=["tournament", "team_id", "player_id", "match_number"], how="left")
    changed = joined["kept"].isna().groupby([joined[k] for k in TEAM_KEYS]).sum().rename("starters_changed")
    out = context[TEAM_KEYS + ["match_number"]].merge(changed.reset_index(), on=TEAM_KEYS, how="left")
    out["starters_changed"] = out["starters_changed"].astype(float).where(out["match_number"] > 1)
    return out[TEAM_KEYS + ["starters_changed"]]


def player_threat(credits: pd.DataFrame) -> pd.DataFrame:
    """Per player per team-match: xg, xa and xg_xa (internal)."""
    xg = credits.groupby(TEAM_KEYS + ["player_id"])["xg"].sum()
    assisted = credits[credits["passer_team_id"].notna() & credits["passer_id"].notna()]
    xa = (assisted.assign(team_id=assisted["passer_team_id"].astype("int64"), player_id=assisted["passer_id"])
          .groupby(TEAM_KEYS + ["player_id"])["xg"].sum())
    out = pd.concat([xg.rename("xg"), xa.rename("xa")], axis=1).fillna(0.0)
    out["xg_xa"] = out["xg"] + out["xa"]
    return out.reset_index()


def top2_share(threat: pd.DataFrame) -> pd.DataFrame:
    """Per team-match: top2_xg_xa_share (NaN when the team's xG + xA is 0)."""
    ranked = threat.sort_values(TEAM_KEYS + ["xg_xa"], ascending=[True, True, True, False])
    top2 = ranked.groupby(TEAM_KEYS).head(2).groupby(TEAM_KEYS)["xg_xa"].sum()
    total = threat.groupby(TEAM_KEYS)["xg_xa"].sum()
    return (top2 / total.where(total > 0)).rename("top2_xg_xa_share").reset_index()


def normalised_entropy(values: pd.Series, n_bins: int = dom.N_ZONES) -> float:
    """Entropy of the values' shares / ln(n_bins): 0 = all in one bin, 1 = equal over all bins."""
    total = values.sum()
    if not total > 0:
        return np.nan
    p = values[values > 0] / total
    return float(np.clip(-(p * np.log(p)).sum() / np.log(n_bins), 0.0, 1.0)) + 0.0


def chance_zone_spread(credits: pd.DataFrame) -> pd.DataFrame:
    """Per team-match: chance_zone_spread over the 18 zones."""
    shots = credits.dropna(subset=["location_x", "location_y"])
    xg = shots.assign(zone=dom.zone_of(shots["location_x"], shots["location_y"]))[TEAM_KEYS + ["zone", "xg"]]
    passes = credits.dropna(subset=["passer_team_id", "pass_x", "pass_y"])
    xa = passes.assign(team_id=passes["passer_team_id"].astype("int64"),
                       zone=dom.zone_of(passes["pass_x"], passes["pass_y"]))[TEAM_KEYS + ["zone", "xg"]]
    by_zone = pd.concat([xg, xa], ignore_index=True).groupby(TEAM_KEYS + ["zone"])["xg"].sum()
    spread = by_zone.groupby(level=TEAM_KEYS).apply(normalised_entropy).rename("chance_zone_spread")
    return spread.reset_index()


def threat_spread(events: pd.DataFrame) -> pd.DataFrame:
    """Per team-match with shots or assists: top2_xg_xa_share and chance_zone_spread."""
    credits = res.shot_credits(events)
    return top2_share(player_threat(credits)).merge(
        chance_zone_spread(credits), on=TEAM_KEYS, how="outer", validate="one_to_one")
