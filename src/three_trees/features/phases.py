"""Phases of play per team-match (supplement addition 2: conversion by phase of play).

Each possession (number within a period, play events only, as in
features.resources.possessions) gets one phase from its first play event, in priority:
set_piece (play pattern From Corner, From Free Kick or From Throw In), counter_attack
(From Counter), transition (is_transition true), settled (control_degree 1 or
is_controlled_possession true), else other. Possessions are credited to the
possession team; shots, xG and OBV count the possession team's own events.

Per team, match and phase: possessions, possession_seconds, shots and xg (non-penalty),
obv (sum of obv_total_net; NaN when the column is missing or empty, e.g. open data),
xg_per_possession, share_of_possessions (of the team's possessions in the match).
Every team-match has a row for each of the five phases.

Limits: periods 1-2 only. Open data has no transition or control fields, so its
possessions fall into set_piece, counter_attack or other.

Output tables: team_match_phase, team_tournament_phase (sums over matches, ratios
recomputed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from three_trees.features import dominance as dom
from three_trees.features import resources as res

PHASES = ("settled", "transition", "counter_attack", "set_piece", "other")
SET_PIECE_PATTERNS = frozenset({"From Corner", "From Free Kick", "From Throw In"})
COUNTER_PATTERN = "From Counter"
POSSESSION_KEYS = ["tournament", "match_id", "period", "possession"]
TEAM_KEYS = res.TEAM_KEYS
SUM_COLUMNS = ["possessions", "possession_seconds", "shots", "xg", "obv"]


def _flag(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].fillna(False).astype(bool) if col in df else pd.Series(False, index=df.index)


def phase_of(first: pd.DataFrame) -> pd.Series:
    """Phase per possession from its first play event's play_pattern, is_transition,
    control_degree and is_controlled_possession."""
    pattern = first["play_pattern"]
    control = first["control_degree"] if "control_degree" in first else pd.Series(np.nan, index=first.index)
    settled = (control == 1) | _flag(first, "is_controlled_possession")
    choice = np.select(
        [pattern.isin(SET_PIECE_PATTERNS), pattern == COUNTER_PATTERN, _flag(first, "is_transition"),
         settled],
        ["set_piece", "counter_attack", "transition", "settled"], default="other")
    return pd.Series(choice, index=first.index)


def possession_phases(events: pd.DataFrame) -> pd.DataFrame:
    """One row per possession: possession team, seconds and phase."""
    poss = res.possessions(events)
    fields = [c for c in ("play_pattern", "is_transition", "control_degree", "is_controlled_possession")
              if c in events]
    first = events[["tournament", "match_id", "index"] + fields].rename(columns={"index": "first_index"})
    poss = poss.merge(first, on=["tournament", "match_id", "first_index"], how="left", validate="one_to_one")
    return poss.assign(phase=phase_of(poss))


def team_match_phase(events: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Per team, match and phase (contract 5.3)."""
    poss = possession_phases(events)
    ev = res.live(events).merge(poss[POSSESSION_KEYS + ["team_id", "phase"]].rename(
        columns={"team_id": "phase_team_id"}), on=POSSESSION_KEYS, how="inner")
    own = ev[ev["team_id"] == ev["phase_team_id"]]
    shots = own[(own["type"] == "Shot") & (own["shot_type"] != "Penalty")]
    keys = TEAM_KEYS + ["phase"]

    df = poss.groupby(keys).agg(possessions=("possession", "size"), possession_seconds=("seconds", "sum"))
    df = df.join(shots.groupby(keys).agg(shots=("id", "size"), xg=("shot_statsbomb_xg", "sum")), how="left")
    has_obv = "obv_total_net" in events and bool(events["obv_total_net"].notna().any())
    if has_obv:
        df = df.join(own.groupby(keys)["obv_total_net"].sum().rename("obv"), how="left")

    sides = dom.match_sides(matches)[TEAM_KEYS + ["team"]]
    grid = sides.merge(pd.DataFrame({"phase": PHASES}), how="cross")
    out = grid.merge(df.reset_index(), on=keys, how="left", validate="one_to_one")
    out[["possessions", "shots"]] = out[["possessions", "shots"]].fillna(0).astype("int64")
    out[["possession_seconds", "xg"]] = out[["possession_seconds", "xg"]].fillna(0.0)
    out["obv"] = out["obv"].fillna(0.0) if has_obv else np.nan
    return _ratios(out, TEAM_KEYS)


def _ratios(df: pd.DataFrame, team_keys: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["xg_per_possession"] = out["xg"] / out["possessions"].where(out["possessions"] > 0)
    total = out.groupby(team_keys)["possessions"].transform("sum")
    out["share_of_possessions"] = out["possessions"] / total.where(total > 0)
    out["phase"] = pd.Categorical(out["phase"], categories=PHASES)
    lead = [c for c in ("tournament", "match_id", "team", "team_id", "matches", "phase") if c in out.columns]
    out = out.sort_values(lead).assign(phase=lambda d: d["phase"].astype(str))
    return out[lead + SUM_COLUMNS + ["xg_per_possession", "share_of_possessions"]].reset_index(drop=True)


def team_tournament_phase(phase_match: pd.DataFrame) -> pd.DataFrame:
    """Sums over the team's matches, ratios recomputed."""
    keys = ["tournament", "team", "team_id", "phase"]
    agg = {c: (c, lambda s: s.sum(min_count=1)) for c in SUM_COLUMNS}
    df = phase_match.groupby(keys, as_index=False).agg(matches=("match_id", "nunique"), **agg)
    df["possessions"] = df["possessions"].astype("int64")
    df["shots"] = df["shots"].astype("int64")
    return _ratios(df, ["tournament", "team_id"])
