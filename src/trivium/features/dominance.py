"""Dominance per team-match as the team's share of the match:
contests (duels, 50/50s, second balls; aerials counted once), territory
(field tilt) and chances (non-penalty xG). Overall dominance = mean of the
three shares. Zone map: the same shares in 18 zones (6 x 3).

share = team value / (team value + opponent value); when both are 0 the
share is 0.5 and <name>_both_zero is True.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trivium.ingest.statsbomb import parse_literal

PITCH_X, PITCH_Y = 120.0, 80.0
FINAL_THIRD_X = 80.0
ZONES_X, ZONES_Y = 6, 3
N_ZONES = ZONES_X * ZONES_Y

DUEL_WON = frozenset({"Won", "Success", "Success In Play", "Success Out"})
DUEL_LOST = frozenset({"Lost In Play", "Lost Out"})
FIFTY_WON = frozenset({"Won", "Success To Team"})
FIFTY_LOST = frozenset({"Lost", "Success To Opposition"})
CONTEST_PARTS = ("duels", "aerials", "fifty_fifties", "second_balls")
TERRITORY_TYPES = frozenset({"Pass", "Carry"})
# Flags on the aerial winner's own event; reported as a check against Aerial Lost, never added.
AERIAL_WON_FLAGS = ("clearance_aerial_won", "miscontrol_aerial_won", "pass_aerial_won", "shot_aerial_won")
MATCH_KEYS = ["tournament", "match_id"]


def share(team: pd.Series, opponent: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(team / (team + opponent), both zero); a share of 0.5 when both are zero."""
    total = team + opponent
    both_zero = total == 0
    return (team / total.where(~both_zero)).where(~both_zero, 0.5), both_zero


def match_sides(matches: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match: tournament, match_id, team, team_id, opponent, opponent_id."""
    home = matches.assign(
        team=matches["home_team"], team_id=matches["home_team_id"],
        opponent=matches["away_team"], opponent_id=matches["away_team_id"],
    )
    away = matches.assign(
        team=matches["away_team"], team_id=matches["away_team_id"],
        opponent=matches["home_team"], opponent_id=matches["home_team_id"],
    )
    cols = MATCH_KEYS + ["team", "team_id", "opponent", "opponent_id"]
    return pd.concat([home[cols], away[cols]], ignore_index=True).sort_values(
        MATCH_KEYS + ["team"]).reset_index(drop=True)


def _with_opponent_id(events: pd.DataFrame, sides: pd.DataFrame) -> pd.DataFrame:
    ev = events[events["period"].isin([1, 2])]
    return ev.merge(sides[MATCH_KEYS + ["team_id", "opponent_id"]], on=MATCH_KEYS + ["team_id"], how="inner")


def fifty_fifty_outcomes(events: pd.DataFrame) -> pd.DataFrame:
    """50/50 events with outcome, credit_self and counted (each contest once).

    StatsBomb records a 50/50 once per player, linked by related_events: one 'Lost'
    plus one Won / Success To Team / Success To Opposition. The paired 'Lost' event
    is not counted; the other event decides the contest. An unpaired event counts.
    """
    ff = events[events["type"] == "50/50"].copy()
    ff["outcome"] = ff["50_50"].map(lambda v: (parse_literal(v) or {}).get("outcome", {}).get("name"))
    ids = set(ff["id"])
    ff["paired"] = ff["related_events"].map(lambda v: any(r in ids for r in (parse_literal(v) or [])))
    ff["credit_self"] = ff["outcome"].isin(FIFTY_WON)
    ff["counted"] = ff["outcome"].isin(FIFTY_WON | FIFTY_LOST) & ~(ff["paired"] & (ff["outcome"] == "Lost"))
    return ff


def contest_events(events: pd.DataFrame, sides: pd.DataFrame) -> pd.DataFrame:
    """One row per contest: part, credited_id (the winning team), event team, location."""
    ev = _with_opponent_id(events, sides)
    duel = ev["type"] == "Duel"
    tackle = duel & (ev["duel_type"] == "Tackle")
    parts = {
        "duels": (tackle & ev["duel_outcome"].isin(DUEL_WON), True),
        "duels_lost": (tackle & ev["duel_outcome"].isin(DUEL_LOST), False),
        "aerials": (duel & (ev["duel_type"] == "Aerial Lost"), False),
        "second_balls": ((ev["type"] == "Ball Recovery") & ~ev["ball_recovery_recovery_failure"], True),
    }
    frames = [ev[mask].assign(part=name.replace("_lost", ""), credit_self=self_)
              for name, (mask, self_) in parts.items()]
    ff = fifty_fifty_outcomes(ev)
    frames.append(ff[ff["counted"]].assign(part="fifty_fifties"))
    cols = MATCH_KEYS + ["team_id", "opponent_id", "part", "credit_self", "location_x", "location_y"]
    out = pd.concat([f[cols] for f in frames], ignore_index=True)
    out["credited_id"] = np.where(out["credit_self"], out["team_id"], out["opponent_id"])
    return out


def territory_events(events: pd.DataFrame, sides: pd.DataFrame) -> pd.DataFrame:
    """Passes and carries (periods 1-2) with location."""
    ev = _with_opponent_id(events, sides)
    cols = MATCH_KEYS + ["team_id", "opponent_id", "location_x", "location_y", "location_bucket"]
    return ev.loc[ev["type"].isin(TERRITORY_TYPES), cols]


def np_xg(events: pd.DataFrame) -> pd.DataFrame:
    """Non-penalty xG (shot_statsbomb_xg) per match and team id, periods 1-2."""
    ev = events[events["period"].isin([1, 2])]
    shots = ev[(ev["type"] == "Shot") & (ev["shot_type"] != "Penalty")]
    return shots.groupby(MATCH_KEYS + ["team_id"])["shot_statsbomb_xg"].sum().rename("np_xg").reset_index()


def _add_shares(df: pd.DataFrame, value: str, opponent: pd.Series, name: str) -> None:
    df[f"{name}_share"], df[f"{name}_both_zero"] = share(df[value], opponent)


def _opponent_values(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """The opponent row's values of cols, aligned to df."""
    opp = df[MATCH_KEYS + ["team_id"] + cols].rename(columns={"team_id": "opponent_id"})
    return df[MATCH_KEYS + ["opponent_id"]].merge(opp, on=MATCH_KEYS + ["opponent_id"], how="left")[cols]


def dominance_team_match(events: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match: contests, territory and chances shares, and dominance."""
    sides = match_sides(matches)
    keys = MATCH_KEYS + ["team_id"]

    contests = contest_events(events, sides)
    won = (contests.groupby(MATCH_KEYS + ["credited_id", "part"]).size().unstack("part")
           .reindex(columns=list(CONTEST_PARTS)).fillna(0).astype("int64")
           .add_suffix("_won").rename_axis(index={"credited_id": "team_id"}).reset_index())
    territory = territory_events(events, sides)
    final_third = (territory[territory["location_x"] >= FINAL_THIRD_X].groupby(keys).size()
                   .rename("final_third_actions").reset_index())

    df = sides.copy()
    for part in (won, final_third, np_xg(events)):
        df = df.merge(part, on=keys, how="left")
    count_cols = [f"{p}_won" for p in CONTEST_PARTS] + ["final_third_actions"]
    df[count_cols] = df[count_cols].fillna(0).astype("int64")
    df["np_xg"] = df["np_xg"].fillna(0.0)
    df["contests_won"] = df[[f"{p}_won" for p in CONTEST_PARTS]].sum(axis=1)

    value_cols = count_cols + ["np_xg", "contests_won"]
    opp = _opponent_values(df, value_cols)
    for p in CONTEST_PARTS:
        _add_shares(df, f"{p}_won", opp[f"{p}_won"], p)
    _add_shares(df, "contests_won", opp["contests_won"], "contests")
    _add_shares(df, "final_third_actions", opp["final_third_actions"], "territory")
    _add_shares(df, "np_xg", opp["np_xg"], "chances")
    df["dominance"] = df[["contests_share", "territory_share", "chances_share"]].mean(axis=1)

    shares = [f"{p}_share" for p in CONTEST_PARTS] + ["contests_share", "territory_share", "chances_share"]
    flags = [s.replace("_share", "_both_zero") for s in shares]
    cols = (MATCH_KEYS + ["team", "team_id", "opponent", "opponent_id"]
            + [f"{p}_won" for p in CONTEST_PARTS] + ["contests_won", "final_third_actions", "np_xg"]
            + shares + ["dominance"] + flags)
    return df[cols]


def zone_of(x: pd.Series, y: pd.Series) -> pd.Series:
    """18 zones, 6 lengthways x 3 widthways, 1-18 (zone_x 0 = own goal end)."""
    zx = np.clip(np.floor(x / (PITCH_X / ZONES_X)), 0, ZONES_X - 1)
    zy = np.clip(np.floor(y / (PITCH_Y / ZONES_Y)), 0, ZONES_Y - 1)
    return (zx * ZONES_Y + zy + 1).astype("int64")


def _views(df: pd.DataFrame) -> pd.DataFrame:
    """Each event twice: from its own team's view (as recorded) and from the
    opponent's view (mirrored: x -> 120 - x, y -> 80 - y). view_id = measured team."""
    own = df.assign(view_id=df["team_id"], x=df["location_x"], y=df["location_y"], own=True)
    other = df.assign(view_id=df["opponent_id"], x=PITCH_X - df["location_x"],
                      y=PITCH_Y - df["location_y"], own=False)
    both = pd.concat([own, other], ignore_index=True)
    both = both[both["x"].notna() & both["y"].notna()]
    return both.assign(zone=zone_of(both["x"], both["y"]))


def team_match_zone(events: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Per team, match and zone (team's view): contests won/lost and share,
    passes + carries of team and opponent and territory share, actions_count
    (contests + passes + carries behind the two shares)."""
    sides = match_sides(matches)
    keys = MATCH_KEYS + ["view_id", "zone"]

    cv = _views(contest_events(events, sides))
    cv["won"] = cv["credited_id"] == cv["view_id"]
    contests = cv.groupby(keys)["won"].agg(contests_won="sum", contests_total="size")
    contests["contests_lost"] = contests["contests_total"] - contests["contests_won"]

    tv = _views(territory_events(events, sides))
    actions = tv.groupby(keys)["own"].agg(team_actions="sum", actions_total="size")
    actions["opponent_actions"] = actions["actions_total"] - actions["team_actions"]

    grid = sides.merge(pd.DataFrame({"zone": range(1, N_ZONES + 1)}), how="cross")
    df = grid.merge(
        contests[["contests_won", "contests_lost"]].join(actions[["team_actions", "opponent_actions"]], how="outer")
        .reset_index().rename(columns={"view_id": "team_id"}),
        on=MATCH_KEYS + ["team_id", "zone"], how="left",
    )
    return _zone_shares(df)


def _zone_shares(df: pd.DataFrame) -> pd.DataFrame:
    counts = ["contests_won", "contests_lost", "team_actions", "opponent_actions"]
    df[counts] = df[counts].fillna(0).astype("int64")
    df["zone_x"] = (df["zone"] - 1) // ZONES_Y
    df["zone_y"] = (df["zone"] - 1) % ZONES_Y
    _add_shares(df, "contests_won", df["contests_lost"], "contests")
    _add_shares(df, "team_actions", df["opponent_actions"], "territory")
    df["actions_count"] = df[counts].sum(axis=1)
    lead = [c for c in MATCH_KEYS + ["team", "team_id", "opponent", "opponent_id", "matches"] if c in df]
    cols = lead + ["zone", "zone_x", "zone_y", "contests_won", "contests_lost", "contests_share",
                   "team_actions", "opponent_actions", "territory_share", "actions_count",
                   "contests_both_zero", "territory_both_zero"]
    return df[cols].reset_index(drop=True)


def team_tournament_zone(zone_match: pd.DataFrame) -> pd.DataFrame:
    """Sums over the team's matches, then shares."""
    counts = ["contests_won", "contests_lost", "team_actions", "opponent_actions"]
    df = (zone_match.groupby(["tournament", "team", "team_id", "zone"], as_index=False)
          .agg(matches=("match_id", "nunique"), **{c: (c, "sum") for c in counts}))
    return _zone_shares(df)


def aerials_vs_hops(events: pd.DataFrame, sides: pd.DataFrame) -> pd.DataFrame:
    """Per team-match: aerials that carry a HOPS win probability, and wins minus that
    probability summed (positive = more aerials won than their difficulty predicts)."""
    ev = _with_opponent_id(events, sides)
    lost = ev[(ev["type"] == "Duel") & (ev["duel_type"] == "Aerial Lost")]
    frames = [lost.assign(won=0.0, prob=lost["duel_duel_win_probability"])]
    for flag in AERIAL_WON_FLAGS:
        if flag in ev:
            prob_col = flag.replace("_aerial_won", "_duel_win_probability")
            w = ev[ev[flag]]
            frames.append(w.assign(won=1.0, prob=w[prob_col] if prob_col in w else np.nan))
    a = pd.concat(frames, ignore_index=True)
    a = a[a["prob"].notna()]
    return (a.assign(diff=a["won"] - a["prob"]).groupby(MATCH_KEYS + ["team_id"])
            .agg(aerials_with_hops=("diff", "size"), aerials_won_above_hops=("diff", "sum")).reset_index())
