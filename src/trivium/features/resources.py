"""Resources, context and chances per team-match from StatsBomb events.

Player minutes: starters from the Starting XI lineup at minute 0; substitutes on at
the Substitution event; time ends at a substitution off, a red card (Foul Committed or
Bad Behaviour with a red or second yellow card), a permanent Player Off, or the last
event of period 2. Minutes are elapsed playing time over periods 1-2 (period 1 length
+ seconds into period 2), so first-half stoppage time is counted once; on_minute and
off_minute are the match clock (minute + second / 60). Jersey number and position come
from the Starting XI and Tactical Shift lineups (substitutes: the first lineup after
they come on), else the player's own events, else the Substitution event; a substitute
in no lineup of the match takes their number from the team's other lineups in the
tournament.

Context and load: match_number and rest_days by team in date order;
starters_minutes_cum (minutes the starting XI played in the team's earlier matches);
players_used (players with minutes > 0).

Chances: np_xg_against (opponent's non-penalty xG); xa (non-penalty xG of shots
assisted by the team's passes: the shot's key pass id, else the pass's assisted shot
id, else an assist-flagged pass linked through related_events); conversion (np_xg /
territory_share); opponent_final_third_entries (opponent completed passes and carries
from x < 80 to x >= 80 in the opponent's coordinates); security (np_xg_against per
entry).

Defensive work: pressures; defensive_actions (Pressure + Duel Tackle + Interception +
Foul Committed + Block); defensive_work_per_dominance (per percentage point of
dominance); ppda (opponent passes at x < 72 in the opponent's coordinates per team
Duel Tackle + Interception + Foul Committed + Block at x >= 48 in the team's
coordinates).

Possession: each possession (number within a period) lasts from its first to its last
play event (period_seconds; lineup, half start / end, substitution, player on / off,
injury stoppage and bad behaviour events left out), credited to the possession team;
possession_minutes, possession_share, pressures_per_opp_possession_min, directness
(mean StatsBomb possession directness over the team's possessions with a value).

Limits: periods 1-2 only (period 5 shootouts are never read). Possession time between
possessions (ball out of play, restarts) is credited to no team.

Shot timeline: shots (penalties included) and Own Goal For events in periods 1-2, team
level (no player ids), with elapsed playing time, half-time, xG (0 for own goals) and
goal flags, for the cumulative xG chart.

Output: columns merged into team_match by features.tables; player_match_minutes
(internal: ids, shirt numbers, positions; no names); shot_timeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trivium.features import dominance as dom
from trivium.ingest.statsbomb import parse_literal

LIVE_PERIODS = (1, 2)
MATCH_KEYS = ["tournament", "match_id"]
TEAM_KEYS = MATCH_KEYS + ["team_id"]
NON_PLAY_TYPES = frozenset({
    "Starting XI", "Half Start", "Half End", "Tactical Shift", "Substitution",
    "Player On", "Player Off", "Injury Stoppage", "Bad Behaviour",
})
LINEUP_TYPES = ("Starting XI", "Tactical Shift")
RED_CARDS = frozenset({"Red Card", "Second Yellow"})
POSITION_GROUPS = ("Goalkeeper", "Defender", "Midfielder", "Forward")
DEFENSIVE_TYPES = ("Pressure", "Interception", "Foul Committed", "Block")
PPDA_OPPONENT_X = 72.0
PPDA_TEAM_X = 48.0
PLAYER_MINUTES_COLUMNS = [
    "tournament", "match_id", "team", "player_id", "jersey_number", "position", "position_group",
    "started", "minutes", "on_minute", "off_minute",
]
SHOT_TIMELINE_COLUMNS = [
    "tournament", "match_id", "team", "team_id", "period", "minute", "second", "elapsed_minute",
    "half_time_minute", "xg", "goal", "penalty", "own_goal",
]


def live(events: pd.DataFrame) -> pd.DataFrame:
    return events[events["period"].isin(LIVE_PERIODS)]


def position_group(position: Any) -> Any:
    """StatsBomb position name -> Goalkeeper / Defender / Midfielder / Forward (None if unknown)."""
    if not isinstance(position, str):
        return None
    if position == "Goalkeeper":
        return "Goalkeeper"
    if "Back" in position:
        return "Defender"
    if "Midfield" in position:
        return "Midfielder"
    if any(word in position for word in ("Wing", "Forward", "Striker")):
        return "Forward"
    return None


def _team_names(events: pd.DataFrame) -> pd.DataFrame:
    return events[TEAM_KEYS + ["team"]].drop_duplicates(TEAM_KEYS)


def period_lengths(events: pd.DataFrame) -> pd.DataFrame:
    """Per match: period_1_s, period_2_s (last event's period_seconds) and match_s."""
    ev = live(events)
    lengths = ev.groupby(MATCH_KEYS + ["period"])["period_seconds"].max().unstack("period")
    lengths = lengths.reindex(columns=list(LIVE_PERIODS)).fillna(0.0)
    lengths.columns = [f"period_{p}_s" for p in LIVE_PERIODS]
    lengths["match_s"] = lengths.sum(axis=1)
    return lengths.reset_index()


def _elapsed_s(df: pd.DataFrame) -> pd.Series:
    """Seconds of playing time before (period, period_seconds); needs period_1_s and match_s."""
    t = np.where(df["period"] == 1, df["period_seconds"], df["period_1_s"] + df["period_seconds"])
    return pd.Series(np.clip(t, 0, df["match_s"]), index=df.index)


def _clock(df: pd.DataFrame) -> pd.Series:
    return df["minute"] + df["second"] / 60.0


def lineup_rows(events: pd.DataFrame) -> pd.DataFrame:
    """One row per player per Starting XI / Tactical Shift lineup: ids, jersey number, position."""
    ev = live(events)
    ev = ev[ev["type"].isin(LINEUP_TYPES) & ev["tactics"].notna()]
    rows = []
    for rec in ev[TEAM_KEYS + ["team", "type", "index", "tactics"]].itertuples(index=False):
        for entry in (parse_literal(rec.tactics) or {}).get("lineup", []):
            rows.append({
                "tournament": rec.tournament, "match_id": rec.match_id, "team_id": rec.team_id,
                "team": rec.team, "type": rec.type, "index": rec.index,
                "player_id": int(entry["player"]["id"]), "jersey_number": entry.get("jersey_number"),
                "position": (entry.get("position") or {}).get("name"),
            })
    cols = TEAM_KEYS + ["team", "type", "index", "player_id", "jersey_number", "position"]
    return pd.DataFrame(rows, columns=cols)


def _off_events(ev: pd.DataFrame) -> pd.DataFrame:
    """Events that end a player's time: substitution off, red card, permanent Player Off."""
    red = (((ev["type"] == "Foul Committed") & ev["foul_committed_card"].isin(RED_CARDS))
           | ((ev["type"] == "Bad Behaviour") & ev["bad_behaviour_card"].isin(RED_CARDS)))
    permanent = (ev["type"] == "Player Off") & ev.get("player_off_permanent", False)
    reason = np.select([ev["type"] == "Substitution", red, permanent],
                       ["substitution", "red card", "player off"], default="")
    out = ev.assign(reason=reason)
    return out[(out["reason"] != "") & out["player_id"].notna()]


def player_match_minutes(events: pd.DataFrame) -> pd.DataFrame:
    """One row per player who started or came on (contract 5.2), plus off_reason (internal)."""
    ev = live(events).merge(period_lengths(events), on=MATCH_KEYS, how="left")
    ev = ev.assign(elapsed_s=_elapsed_s(ev), clock=_clock(ev))
    lineups = lineup_rows(events)

    starters = lineups[lineups["type"] == "Starting XI"].drop_duplicates(TEAM_KEYS + ["player_id"])
    starters = starters.assign(started=True, on_s=0.0, on_minute=0.0, on_index=0)
    subs = ev[ev["type"] == "Substitution"]
    subs_on = subs[subs["substitution_replacement_id"].notna()].assign(
        player_id=subs["substitution_replacement_id"].astype("int64"), started=False,
        on_s=subs["elapsed_s"], on_minute=subs["clock"], on_index=subs["index"],
        sub_position=subs["position"])
    on_cols = TEAM_KEYS + ["team", "player_id", "started", "on_s", "on_minute", "on_index"]
    players = pd.concat([starters[on_cols], subs_on[on_cols + ["sub_position"]]], ignore_index=True)
    players = players.drop_duplicates(TEAM_KEYS + ["player_id"], keep="first")

    off = _off_events(ev).assign(player_id=lambda d: d["player_id"].astype("int64"))
    off = off.merge(players[TEAM_KEYS + ["player_id", "on_s"]], on=TEAM_KEYS + ["player_id"], how="inner")
    off = off[off["elapsed_s"] >= off["on_s"]].sort_values("elapsed_s", kind="stable")
    first_off = off.drop_duplicates(TEAM_KEYS + ["player_id"], keep="first")[
        TEAM_KEYS + ["player_id", "elapsed_s", "clock", "reason"]].rename(
        columns={"elapsed_s": "off_s", "clock": "off_minute", "reason": "off_reason"})

    df = players.merge(first_off, on=TEAM_KEYS + ["player_id"], how="left", validate="one_to_one")
    df = df.merge(period_lengths(events)[MATCH_KEYS + ["match_s"]], on=MATCH_KEYS, how="left")
    df["minutes"] = (df["off_s"].fillna(df["match_s"]) - df["on_s"]).clip(lower=0) / 60.0
    df["off_reason"] = df["off_reason"].fillna("")
    df = _add_jersey_and_position(df, lineups, ev)
    df["position_group"] = df["position"].map(position_group)
    df["started"] = df["started"].astype(bool)
    df["jersey_number"] = pd.to_numeric(df["jersey_number"], errors="coerce").astype("Int64")
    df["player_id"] = df["player_id"].astype("int64")
    return (df.sort_values(MATCH_KEYS + ["team", "started", "on_minute", "player_id"],
                           ascending=[True, True, True, False, True, True])
            [PLAYER_MINUTES_COLUMNS + ["team_id", "off_reason"]].reset_index(drop=True))


def _add_jersey_and_position(df: pd.DataFrame, lineups: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Starters: Starting XI. Substitutes: first lineup after coming on, else own events, else the
    Substitution event's position. Jersey number from any lineup of the match, else the player's
    most frequent number in the team's lineups of the tournament (squad numbers are fixed)."""
    keys = TEAM_KEYS + ["player_id"]
    numbered = lineups.dropna(subset=["jersey_number"])
    jersey = numbered.drop_duplicates(keys)[keys + ["jersey_number"]]
    squad_keys = ["tournament", "team_id", "player_id"]
    squad = (numbered.groupby(squad_keys)["jersey_number"].agg(lambda s: s.mode().iloc[0])
             .rename("squad_number").reset_index())
    after = lineups.merge(df[keys + ["on_index", "started"]], on=keys, how="inner")
    after = after[(after["started"] & (after["type"] == "Starting XI"))
                  | (~after["started"] & (after["index"] >= after["on_index"]))]
    from_lineup = (after.dropna(subset=["position"]).sort_values("index", kind="stable")
                   .drop_duplicates(keys)[keys + ["position"]]
                   .rename(columns={"position": "lineup_position"}))
    own = ev[ev["player_id"].notna() & ev["position"].notna()].assign(
        player_id=lambda d: d["player_id"].astype("int64"))
    from_events = (own.sort_values("index", kind="stable").drop_duplicates(keys)[keys + ["position"]]
                   .rename(columns={"position": "event_position"}))
    out = (df.merge(jersey, on=keys, how="left", validate="one_to_one")
           .merge(squad, on=squad_keys, how="left", validate="many_to_one")
           .merge(from_lineup, on=keys, how="left", validate="one_to_one")
           .merge(from_events, on=keys, how="left", validate="one_to_one"))
    out["jersey_number"] = out["jersey_number"].fillna(out["squad_number"])
    out["position"] = out["lineup_position"].fillna(out["event_position"]).fillna(out["sub_position"])
    return out


def match_context(matches: pd.DataFrame) -> pd.DataFrame:
    """Per team-match: match_number (1 = first) and rest_days (NaN for match 1), by date order."""
    sides = dom.match_sides(matches).merge(
        matches[MATCH_KEYS + ["match_date", "kick_off"]], on=MATCH_KEYS, how="left", validate="many_to_one")
    sides = sides.sort_values(["tournament", "team", "match_date", "kick_off", "match_id"], kind="stable")
    dates = pd.to_datetime(sides["match_date"])
    by_team = sides.groupby(["tournament", "team"], sort=False)
    sides["match_number"] = by_team.cumcount() + 1
    gaps = dates.groupby([sides["tournament"], sides["team"]], sort=False).diff()
    sides["rest_days"] = gaps.dt.days.astype(float)
    return sides[TEAM_KEYS + ["team", "match_number", "rest_days"]].reset_index(drop=True)


def _earlier_minutes(minutes: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Each team-match's starters joined to their minutes in the team's earlier matches."""
    pm = minutes.merge(context[TEAM_KEYS + ["match_number"]], on=TEAM_KEYS, how="left",
                       validate="many_to_one")
    starters = pm[pm["started"]][TEAM_KEYS + ["player_id", "match_number"]]
    history = pm[["tournament", "team_id", "player_id", "match_number", "minutes"]].rename(
        columns={"match_number": "earlier_number"})
    joined = starters.merge(history, on=["tournament", "team_id", "player_id"], how="left")
    joined = joined[joined["earlier_number"] < joined["match_number"]]
    return joined


def squad_load(minutes: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Per team-match: players_used and starters_minutes_cum."""
    used = (minutes[minutes["minutes"] > 0].groupby(TEAM_KEYS)["player_id"].nunique()
            .rename("players_used"))
    earlier = _earlier_minutes(minutes, context)
    cum = earlier.groupby(TEAM_KEYS)["minutes"].sum().rename("starters_minutes_cum")
    out = context[TEAM_KEYS].merge(used.reset_index(), on=TEAM_KEYS, how="left")
    out = out.merge(cum.reset_index(), on=TEAM_KEYS, how="left")
    out["players_used"] = out["players_used"].fillna(0).astype("int64")
    out["starters_minutes_cum"] = out["starters_minutes_cum"].fillna(0.0)
    return out


def _pass_links(ev: pd.DataFrame) -> pd.DataFrame:
    """shot_id -> pass_id: shot key pass id, else pass assisted shot id, else assist-flagged
    passes whose related_events hold the shot."""
    shots = ev[ev["type"] == "Shot"]
    passes = ev[ev["type"] == "Pass"]
    links = [shots.loc[shots["shot_key_pass_id"].notna(), ["id", "shot_key_pass_id"]]
             .set_axis(["shot_id", "pass_id"], axis=1)]
    if "pass_assisted_shot_id" in passes:
        links.append(passes.loc[passes["pass_assisted_shot_id"].notna(), ["pass_assisted_shot_id", "id"]]
                     .set_axis(["shot_id", "pass_id"], axis=1))
    flags = [c for c in ("pass_shot_assist", "pass_goal_assist") if c in passes]
    flagged = passes[passes[flags].any(axis=1)] if flags else passes.iloc[0:0]
    shot_ids = set(shots["id"])
    related = [(sid, pid) for pid, rel in zip(flagged["id"], flagged["related_events"])
               for sid in (parse_literal(rel) or []) if sid in shot_ids]
    links.append(pd.DataFrame(related, columns=["shot_id", "pass_id"]))
    out = pd.concat(links, ignore_index=True).astype("string")
    return out.drop_duplicates("shot_id", keep="first")


def shot_credits(events: pd.DataFrame) -> pd.DataFrame:
    """Non-penalty shots in periods 1-2: shooter team / player, xG and location, and the
    key pass's team / player and location (NaN when unassisted)."""
    ev = live(events)
    shots = ev[(ev["type"] == "Shot") & (ev["shot_type"] != "Penalty")]
    shots = shots[TEAM_KEYS + ["id", "player_id", "shot_statsbomb_xg", "location_x", "location_y"]].rename(
        columns={"id": "shot_id", "shot_statsbomb_xg": "xg"})
    shots = shots.assign(shot_id=shots["shot_id"].astype("string"), xg=shots["xg"].fillna(0.0))
    passes = ev[ev["type"] == "Pass"][["id", "team_id", "player_id", "location_x", "location_y"]].rename(
        columns={"id": "pass_id", "team_id": "passer_team_id", "player_id": "passer_id",
                 "location_x": "pass_x", "location_y": "pass_y"})
    passes = passes.assign(pass_id=passes["pass_id"].astype("string"))
    out = shots.merge(_pass_links(ev), on="shot_id", how="left", validate="one_to_one")
    return out.merge(passes, on="pass_id", how="left", validate="many_to_one")


def shot_timeline(events: pd.DataFrame) -> pd.DataFrame:
    """Shots and Own Goal For events (credited to the benefiting team), periods 1-2, in match order:
    elapsed_minute (first-half stoppage counted once), half_time_minute, xg, goal, penalty, own_goal."""
    ev = live(events)
    ev = ev[ev["type"].isin(["Shot", "Own Goal For"])].merge(period_lengths(events), on=MATCH_KEYS, how="left")
    shot = ev["type"] == "Shot"
    out = ev.assign(
        elapsed_minute=_elapsed_s(ev) / 60.0, half_time_minute=ev["period_1_s"] / 60.0,
        xg=ev["shot_statsbomb_xg"].where(shot, 0.0).fillna(0.0),
        goal=(shot & (ev["shot_outcome"] == "Goal")) | ~shot,
        penalty=shot & (ev["shot_type"] == "Penalty"), own_goal=~shot)
    out = out.sort_values(MATCH_KEYS + ["index"], kind="stable")
    return out[SHOT_TIMELINE_COLUMNS].astype({"goal": bool, "penalty": bool, "own_goal": bool}).reset_index(drop=True)


def chances(events: pd.DataFrame, team_match: pd.DataFrame) -> pd.DataFrame:
    """Per team-match: np_xg_against, xa, conversion, opponent_final_third_entries, security.
    team_match needs TEAM_KEYS, opponent_id, np_xg and territory_share."""
    ev = live(events)
    credits = shot_credits(events)
    xa = (credits[credits["passer_team_id"].notna()]
          .assign(team_id=lambda d: d["passer_team_id"].astype("int64"))
          .groupby(TEAM_KEYS)["xg"].sum().rename("xa"))
    move = ev[ev["type"].isin(["Pass", "Carry"])]
    end_x = move["pass_end_location_x"].where(move["type"] == "Pass", move["carry_end_location_x"])
    completed = (move["type"] == "Carry") | move["pass_outcome"].isna()
    entry = completed & (move["location_x"] < dom.FINAL_THIRD_X) & (end_x >= dom.FINAL_THIRD_X)
    entries = (move[entry].groupby(TEAM_KEYS).size().rename("opponent_final_third_entries")
               .rename_axis(index={"team_id": "opponent_id"}))

    df = team_match[TEAM_KEYS + ["opponent_id", "np_xg", "territory_share"]].copy()
    opp_xg = df[TEAM_KEYS + ["np_xg"]].rename(columns={"team_id": "opponent_id", "np_xg": "np_xg_against"})
    df = df.merge(opp_xg, on=MATCH_KEYS + ["opponent_id"], how="left", validate="one_to_one")
    df = df.merge(xa.reset_index(), on=TEAM_KEYS, how="left", validate="one_to_one")
    df = df.merge(entries.reset_index(), on=MATCH_KEYS + ["opponent_id"], how="left", validate="one_to_one")
    df["xa"] = df["xa"].fillna(0.0)
    df["np_xg_against"] = df["np_xg_against"].fillna(0.0)
    df["opponent_final_third_entries"] = df["opponent_final_third_entries"].fillna(0).astype("int64")
    df["conversion"] = df["np_xg"] / df["territory_share"].where(df["territory_share"] > 0)
    df["security"] = df["np_xg_against"] / df["opponent_final_third_entries"].where(
        df["opponent_final_third_entries"] > 0)
    return df[TEAM_KEYS + ["np_xg_against", "xa", "conversion", "opponent_final_third_entries", "security"]]


def _defensive_mask(ev: pd.DataFrame, with_pressures: bool) -> pd.Series:
    types = DEFENSIVE_TYPES if with_pressures else tuple(t for t in DEFENSIVE_TYPES if t != "Pressure")
    return ev["type"].isin(types) | ((ev["type"] == "Duel") & (ev["duel_type"] == "Tackle"))


def defensive_work(events: pd.DataFrame, team_match: pd.DataFrame) -> pd.DataFrame:
    """Per team-match: pressures, defensive_actions, defensive_work_per_dominance, ppda.
    team_match needs TEAM_KEYS, opponent_id and dominance."""
    ev = live(events)
    pressures = ev[ev["type"] == "Pressure"].groupby(TEAM_KEYS).size().rename("pressures")
    actions = ev[_defensive_mask(ev, True)].groupby(TEAM_KEYS).size().rename("defensive_actions")
    high = ev[_defensive_mask(ev, False) & (ev["location_x"] >= PPDA_TEAM_X)]
    high_actions = high.groupby(TEAM_KEYS).size().rename("ppda_actions")
    opp_passes = (ev[(ev["type"] == "Pass") & (ev["location_x"] < PPDA_OPPONENT_X)].groupby(TEAM_KEYS).size()
                  .rename("ppda_passes").rename_axis(index={"team_id": "opponent_id"}))

    df = team_match[TEAM_KEYS + ["opponent_id", "dominance"]].copy()
    for part in (pressures, actions, high_actions):
        df = df.merge(part.reset_index(), on=TEAM_KEYS, how="left", validate="one_to_one")
    df = df.merge(opp_passes.reset_index(), on=MATCH_KEYS + ["opponent_id"], how="left",
                  validate="one_to_one")
    counts = ["pressures", "defensive_actions", "ppda_actions", "ppda_passes"]
    df[counts] = df[counts].fillna(0).astype("int64")
    points = (100 * df["dominance"]).where(df["dominance"] > 0)
    df["defensive_work_per_dominance"] = df["defensive_actions"] / points
    df["ppda"] = df["ppda_passes"] / df["ppda_actions"].where(df["ppda_actions"] > 0)
    return df[TEAM_KEYS + ["pressures", "defensive_actions", "defensive_work_per_dominance", "ppda"]]


def possessions(events: pd.DataFrame) -> pd.DataFrame:
    """One row per possession (match, period, possession number) with play events: possession
    team, start_s, end_s, seconds and the first play event's index."""
    ev = live(events)
    ev = ev[~ev["type"].isin(NON_PLAY_TYPES)]
    keys = MATCH_KEYS + ["period", "possession"]
    out = ev.sort_values(MATCH_KEYS + ["index"], kind="stable").groupby(keys, sort=False).agg(
        team_id=("possession_team_id", "first"), start_s=("period_seconds", "min"),
        end_s=("period_seconds", "max"), first_index=("index", "first"))
    out["seconds"] = out["end_s"] - out["start_s"]
    return out.reset_index()


def possession_measures(events: pd.DataFrame, team_match: pd.DataFrame) -> pd.DataFrame:
    """Per team-match: possession_minutes, possession_share, pressures_per_opp_possession_min,
    directness. team_match needs TEAM_KEYS, opponent_id and pressures."""
    poss = possessions(events)
    ev = live(events)
    keys = MATCH_KEYS + ["period", "possession"]
    if "possession_directness" in ev:
        direct = ev.dropna(subset=["possession_directness"]).groupby(keys)["possession_directness"].first()
        poss = poss.merge(direct.reset_index(), on=keys, how="left", validate="one_to_one")
    else:
        poss["possession_directness"] = np.nan
    team = poss.groupby(TEAM_KEYS).agg(possession_s=("seconds", "sum"),
                                       directness=("possession_directness", "mean"))

    df = team_match[TEAM_KEYS + ["opponent_id", "pressures"]].merge(
        team.reset_index(), on=TEAM_KEYS, how="left", validate="one_to_one")
    df["possession_minutes"] = df["possession_s"].fillna(0.0) / 60.0
    opp = df[TEAM_KEYS + ["possession_minutes"]].rename(
        columns={"team_id": "opponent_id", "possession_minutes": "opp_minutes"})
    df = df.merge(opp, on=MATCH_KEYS + ["opponent_id"], how="left", validate="one_to_one")
    df["possession_share"], _ = dom.share(df["possession_minutes"], df["opp_minutes"].fillna(0.0))
    df["pressures_per_opp_possession_min"] = df["pressures"] / df["opp_minutes"].where(df["opp_minutes"] > 0)
    return df[TEAM_KEYS + ["possession_minutes", "possession_share", "pressures_per_opp_possession_min",
                           "directness"]]
