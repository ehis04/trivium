"""Features stage: assembles the feature tables from dominance, resources, spread
and phases.

team_match: one row per team per match with dominance and its parts
(features.dominance), aerials vs HOPS, chances, conversion and security, defensive
work and pressing, possession and directness, context and load
(features.resources), load and threat spread (features.spread). When the tracking
stage has written team_match_provider (men), its columns are merged in on (tournament,
match_id, team) and hsr_without_ball_per_dominance = hsr_without_ball_km * 1000 /
(dominance * 100) is computed; without it those columns are present and empty. The
tracking columns (TRACKING_COLUMNS) are left empty for matches where more than 30% of
player positions were filled in by the provider (tracking.running.excluded_matches);
FIFA physical totals and FIFA phase shares are kept for them.

team_tournament: per team, matches and the per-match mean of every numeric
team_match measure (ids and both-zero flags left out).

The efficiency and profile tables (efficiency_team_match,
efficiency_team_tournament, efficiency_position_group, profile_by_stage,
profile_by_overperformance) are written by the profiles stage, not by features.

Output tables: team_match, team_tournament, team_match_zone, team_tournament_zone,
team_match_phase, team_tournament_phase, player_match_minutes (internal),
shot_timeline, formation_listed (features.formations), checks_features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from three_trees.config import Config
from three_trees.features import dominance as dom
from three_trees.features.formations import listed_formations
from three_trees.features import phases as ph
from three_trees.features import resources as res
from three_trees.features import spread as sp
from three_trees.storage import read_table, table_exists, write_table

TEAM_MATCH_ROWS = {"men": 208, "women": 104}
SHARE_COLUMNS = (
    "duels_share", "aerials_share", "fifty_fifties_share", "second_balls_share",
    "contests_share", "territory_share", "chances_share", "dominance",
)
ON_TARGET = frozenset({"Goal", "Saved", "Saved to Post"})
# FIFA figures for Ireland v Panama from the published match report (Ireland first).
IRELAND_PANAMA = ("Republic of Ireland", "Panama")
FIFA_DECK = {
    "pressures": "198 v 143",
    "attempts (on target)": "18 (8) v 13 (4)",
    "possession %": "33.4 v 53.8",
    "distance km": "109.5 v 98.8",
}
TEAM_KEYS = res.TEAM_KEYS
PROVIDER_KEYS = ["tournament", "match_id", "team"]
FIFA_PHASES = (
    "build_up", "progression", "final_third", "long_ball", "attacking_transition",
    "counter_attack", "press", "counter_press", "defensive_transition", "recovery", "block",
)
PROVIDER_COLUMNS = (
    "distance_km", "hsr_km", "sprint_km", "distance_km_tracking", "hsr_km_tracking",
    "hsr_with_ball_km", "hsr_without_ball_km", "hsr_out_of_play_km", "hsr_without_ball_defensive_third_km",
    "top3_hsr_share", "line_height_m", "team_length_m", "team_width_m",
    *(f"fifa_{p}_share" for p in FIFA_PHASES),
)
EVENT_COLUMNS = (
    "np_xg_against", "xa", "conversion", "opponent_final_third_entries", "security",
    "pressures", "defensive_actions", "defensive_work_per_dominance", "ppda",
    "possession_minutes", "possession_share", "pressures_per_opp_possession_min", "directness",
    "top2_xg_xa_share", "chance_zone_spread", "players_used", "minutes_outside_top11_share",
    "starters_changed", "starters_minutes_cum", "match_number", "rest_days",
)
TRACKING_COLUMNS = (
    "distance_km_tracking", "hsr_km_tracking", "hsr_with_ball_km", "hsr_without_ball_km", "hsr_out_of_play_km",
    "hsr_without_ball_defensive_third_km", "top3_hsr_share", "line_height_m", "team_length_m", "team_width_m",
)
NEW_COLUMNS = EVENT_COLUMNS + ("hsr_without_ball_per_dominance",) + PROVIDER_COLUMNS
# Coverage rule per column: "all" (every team-match), "after_first" (all but each team's
# first match), "defined" (NaN only where the ratio is undefined; listed), "men" (tracking).
COVERAGE = {c: "all" for c in EVENT_COLUMNS} | {
    "starters_changed": "after_first", "rest_days": "after_first",
    "conversion": "defined", "security": "defined", "ppda": "defined",
    "pressures_per_opp_possession_min": "defined", "top2_xg_xa_share": "defined",
    "chance_zone_spread": "defined", "hsr_without_ball_per_dominance": "men",
} | {c: "men" for c in PROVIDER_COLUMNS}
UNIT_SHARE_COLUMNS = (
    "possession_share", "directness", "top2_xg_xa_share", "chance_zone_spread",
    "minutes_outside_top11_share", "top3_hsr_share", *(f"fifa_{p}_share" for p in FIFA_PHASES),
)
TEAM_TOURNAMENT_ROWS = {"men": 48, "women": 24}
ID_COLUMNS = frozenset({"match_id", "team_id", "opponent_id"})
STARTERS = 11
TEAM_MINUTES_TOLERANCE = 0.01
DEFINITIONS = {
    "pressures": "Pressure events of the team, periods 1-2",
    "defensive_actions": "Pressure + Duel (Tackle, any outcome) + Interception + Foul Committed + Block "
                         "events of the team, periods 1-2",
    "ppda": "opponent Pass events (any outcome) starting at x < 72 in the opponent's coordinates / "
            "team Duel (Tackle) + Interception + Foul Committed + Block events at x >= 48 in the team's "
            "coordinates",
    "possession": "each possession number within a period, from its first to its last play event "
                  "(period_seconds), credited to the possession team; Starting XI, Half Start, Half End, "
                  "Tactical Shift, Substitution, Player On, Player Off, Injury Stoppage, Bad Behaviour "
                  "left out",
}


def _check(rows: list, check: str, target: Any, value: Any, passed: bool | None) -> None:
    rows.append({"check": check, "target": str(target), "value": str(value),
                 "passed": pd.NA if passed is None else bool(passed)})


def _ireland_panama(rows: list, events: pd.DataFrame, matches: pd.DataFrame, team_match: pd.DataFrame) -> None:
    ire, pan = IRELAND_PANAMA
    m = matches[(matches["tournament"] == "men")
                & matches["home_team"].isin(IRELAND_PANAMA) & matches["away_team"].isin(IRELAND_PANAMA)]
    if m.empty:
        _check(rows, "men: Ireland v Panama found", "yes", "no", False)
        return
    match_id = int(m["match_id"].iloc[0])
    ev = events[(events["tournament"] == "men") & (events["match_id"] == match_id) & events["period"].isin([1, 2])]
    tm = team_match[(team_match["tournament"] == "men") & (team_match["match_id"] == match_id)].set_index("team")

    def per_team(mask: pd.Series) -> tuple[int, int]:
        counts = ev.loc[mask.astype(bool), "team"].value_counts()
        return int(counts.get(ire, 0)), int(counts.get(pan, 0))

    pressures = per_team((ev["type"] == "Pressure").astype(int))
    shots = per_team((ev["type"] == "Shot").astype(int))
    on_target = per_team(((ev["type"] == "Shot") & ev["shot_outcome"].isin(ON_TARGET)).astype(int))
    passes = per_team((ev["type"] == "Pass").astype(int))
    pass_share = [100 * p / max(sum(passes), 1) for p in passes]

    label = "men: Ireland v Panama (FIFA deck target, StatsBomb value)"
    _check(rows, f"{label}: pressures", FIFA_DECK["pressures"], f"{pressures[0]} v {pressures[1]}", None)
    _check(rows, f"{label}: attempts (on target)", FIFA_DECK["attempts (on target)"],
           f"{shots[0]} ({on_target[0]}) v {shots[1]} ({on_target[1]})", None)
    _check(rows, f"{label}: possession % (StatsBomb pass share)", FIFA_DECK["possession %"],
           f"{pass_share[0]:.1f} v {pass_share[1]:.1f}", None)
    for col in ("contests_share", "territory_share", "chances_share", "dominance"):
        _check(rows, f"men: Ireland v Panama {col} (StatsBomb)", "no FIFA figure",
               f"{tm.loc[ire, col]:.3f} v {tm.loc[pan, col]:.3f}", None)


def check_features(events: pd.DataFrame, matches: pd.DataFrame, team_match: pd.DataFrame,
                   zone_match: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    sides = dom.match_sides(matches)
    for t in sorted(matches["tournament"].unique()):
        ev = events[(events["tournament"] == t) & events["period"].isin([1, 2])]
        m = matches[matches["tournament"] == t]
        tm = team_match[team_match["tournament"] == t]
        zm = zone_match[zone_match["tournament"] == t]
        target = TEAM_MATCH_ROWS.get(t)

        _check(rows, f"{t}: team_match rows", target, len(tm), len(tm) == target)
        for col in SHARE_COLUMNS:
            sums = tm.groupby("match_id")[col].sum()
            ok = int(np.isclose(sums, 1.0).sum())
            _check(rows, f"{t}: {col} of the two teams sums to 1", f"{len(m)} of {len(m)}",
                   f"{ok} of {len(sums)}", ok == len(m))
        flagged = {c: int(tm[c].sum()) for c in tm.columns if c.endswith("_both_zero") and tm[c].any()}
        _check(rows, f"{t}: team-match rows with both values 0 (share set to 0.5)", "listed", flagged or "none", None)

        home = m[["match_id", "home_team_id", "home_score", "away_score"]].rename(columns={"home_team_id": "team_id"})
        away = m[["match_id", "away_team_id", "away_score", "home_score"]].rename(
            columns={"away_team_id": "team_id", "away_score": "home_score", "home_score": "away_score"})
        gd = pd.concat([home, away]).assign(goal_difference=lambda d: d["home_score"] - d["away_score"])
        merged = tm.merge(gd[["match_id", "team_id", "goal_difference"]], on=["match_id", "team_id"])
        for col in ("dominance", "contests_share", "territory_share", "chances_share"):
            r = merged[col].corr(merged["goal_difference"])
            _check(rows, f"{t}: correlation of {col} with goal difference", "> 0", f"{r:.3f}",
                   r > 0 if col == "dominance" else None)

        aerial_lost = int(((ev["type"] == "Duel") & (ev["duel_type"] == "Aerial Lost")).sum())
        aerial_flags = int(sum(ev[f].sum() for f in dom.AERIAL_WON_FLAGS if f in ev))
        _check(rows, f"{t}: *_aerial_won flags = Aerial Lost duels (aerials counted once)",
               aerial_lost, aerial_flags, aerial_flags == aerial_lost)

        ff = dom.fifty_fifty_outcomes(ev)
        pairs = int((ff["paired"] & (ff["outcome"] == "Lost")).sum())
        conflict = int((ff["paired"] & (ff["outcome"] == "Success To Opposition")).sum())
        _check(rows, f"{t}: 50/50 events recorded once per player (events, paired 'Lost' dropped, counted)",
               "each contest once", f"{len(ff)}, {pairs}, {int(ff['counted'].sum())}",
               int(ff["counted"].sum()) == len(ff) - pairs)
        _check(rows, f"{t}: 50/50 pairs where 'Lost' meets 'Success To Opposition' (credited to the 'Lost' side)",
               "listed", conflict, None)

        terr = dom.territory_events(ev, sides)
        located = terr[terr["location_x"].notna()]
        agree = ((located["location_x"] >= dom.FINAL_THIRD_X) == (located["location_bucket"] == "final_third")).mean()
        _check(rows, f"{t}: final third x >= 80 agrees with location_bucket (passes + carries)",
               "report", f"{100 * agree:.1f}%", None)

        _check(rows, f"{t}: team_match_zone rows", target * dom.N_ZONES if target else None, len(zm),
               len(zm) == (target or 0) * dom.N_ZONES)
        zone_actions = zm.groupby(["match_id", "team_id"])["team_actions"].sum()
        match_actions = located.groupby(["match_id", "team_id"]).size().reindex(zone_actions.index, fill_value=0)
        ok = int((zone_actions == match_actions).sum())
        _check(rows, f"{t}: zone team_actions sum to the team's located passes + carries",
               f"{len(zone_actions)} of {len(zone_actions)}", f"{ok} of {len(zone_actions)}",
               ok == len(zone_actions))
        mirrored = zm.merge(
            zm.assign(zone=dom.N_ZONES + 1 - zm["zone"])[["match_id", "team_id", "zone", "team_actions"]]
            .rename(columns={"team_id": "opponent_id", "team_actions": "mirror_actions"}),
            on=["match_id", "opponent_id", "zone"])
        moved = (mirrored["opponent_actions"] - mirrored["mirror_actions"]).abs().sum() / 2
        agree = 1 - moved / max(mirrored["opponent_actions"].sum(), 1)
        _check(rows, f"{t}: opponent actions in the same zone after mirroring as in the opponent's own view "
                     "(differs only for actions exactly on a zone line, e.g. kick-offs at x = 60)",
               ">= 98%", f"{100 * agree:.1f}%", agree >= 0.98)

        hops = dom.aerials_vs_hops(ev, sides)
        with_prob = int(((ev["type"] == "Duel") & (ev["duel_type"] == "Aerial Lost")
                         & ev["duel_duel_win_probability"].notna()).sum())
        per_match = hops.groupby("match_id")["aerials_won_above_hops"].sum().abs().max()
        _check(rows, f"{t}: aerials with HOPS win probability (both teams' events carry one, "
                     "so won minus probability sums to 0 in every match)",
               "about 1.9k (both tournaments)",
               f"{with_prob} aerials; largest match sum {per_match:.1e}; "
               f"team-match sd {hops['aerials_won_above_hops'].std():.2f}", None)
    _ireland_panama(rows, events, matches, team_match)
    return pd.DataFrame(rows, columns=["check", "target", "value", "passed"]).astype({"passed": "boolean"})


def merge_provider(team_match: pd.DataFrame, provider_table: pd.DataFrame | None,
              excluded: pd.DataFrame | None = None) -> pd.DataFrame:
    """team_match_provider columns (men) merged in, TRACKING_COLUMNS emptied for the excluded matches
    (tournament, match_id), hsr_without_ball_per_dominance computed; without team_match_provider every
    column of PROVIDER_COLUMNS is present and empty."""
    tm = team_match.drop(columns=[c for c in PROVIDER_COLUMNS if c in team_match.columns])
    if provider_table is None:
        tm = tm.assign(**{c: np.nan for c in PROVIDER_COLUMNS})
    else:
        cols = [c for c in provider_table.columns if c not in PROVIDER_KEYS and c not in tm.columns]
        tm = tm.merge(provider_table[PROVIDER_KEYS + cols], on=PROVIDER_KEYS, how="left", validate="one_to_one")
        tm = tm.assign(**{c: np.nan for c in PROVIDER_COLUMNS if c not in tm.columns})
    if excluded is not None and len(excluded):
        out = pd.MultiIndex.from_frame(tm[["tournament", "match_id"]]).isin(
            pd.MultiIndex.from_frame(excluded[["tournament", "match_id"]]))
        tm.loc[out, list(TRACKING_COLUMNS)] = np.nan
    tm["hsr_without_ball_per_dominance"] = (
        tm["hsr_without_ball_km"] * 1000 / (100 * tm["dominance"]).where(tm["dominance"] > 0))
    return tm


def build_team_match(events: pd.DataFrame, matches: pd.DataFrame, provider_table: pd.DataFrame | None = None,
                     excluded: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(team_match, player_match_minutes)."""
    tm = dom.dominance_team_match(events, matches)
    hops = dom.aerials_vs_hops(events, dom.match_sides(matches))
    tm = tm.merge(hops, on=TEAM_KEYS, how="left", validate="one_to_one")
    tm["aerials_with_hops"] = tm["aerials_with_hops"].fillna(0).astype("int64")
    tm["aerials_won_above_hops"] = tm["aerials_won_above_hops"].fillna(0.0)

    minutes = res.player_match_minutes(events)
    context = res.match_context(matches)
    tm = tm.merge(context.drop(columns="team"), on=TEAM_KEYS, how="left", validate="one_to_one")
    for part in (res.chances(events, tm), res.defensive_work(events, tm)):
        tm = tm.merge(part, on=TEAM_KEYS, how="left", validate="one_to_one")
    parts = (res.possession_measures(events, tm), sp.threat_spread(events), res.squad_load(minutes, context),
             sp.minutes_outside_top11(minutes), sp.starters_changed(minutes, context))
    for part in parts:
        tm = tm.merge(part, on=TEAM_KEYS, how="left", validate="one_to_one")
    tm = merge_provider(tm, provider_table, excluded)
    return tm, minutes


def team_tournament(team_match: pd.DataFrame) -> pd.DataFrame:
    """Per team: matches and the per-match mean of every numeric team_match measure."""
    types = pd.api.types
    numeric = [c for c in team_match.columns if c not in ID_COLUMNS
               and types.is_numeric_dtype(team_match[c]) and not types.is_bool_dtype(team_match[c])]
    grouped = team_match.groupby(["tournament", "team"], as_index=False)
    head = grouped.agg(team_id=("team_id", "first"), matches=("match_id", "nunique"))
    means = grouped[numeric].mean()
    return head.merge(means, on=["tournament", "team"], validate="one_to_one")


def _summary(series: pd.Series) -> str:
    s = series.astype(float)
    if s.notna().sum() == 0:
        return f"n {len(s)}, no values"
    return (f"n {len(s)}, mean {s.mean():.3f}, min {s.min():.3f}, max {s.max():.3f}, "
            f"null {100 * s.isna().mean():.1f}%")


def provider_expected(team_match: pd.DataFrame, provider_table: pd.DataFrame | None,
                 excluded: pd.DataFrame | None) -> pd.DataFrame | None:
    """Per team_match row and provider column: True where a value is expected (team_match_provider has one and,
    for TRACKING_COLUMNS, the match is not excluded); None when team_match_provider is not built."""
    if provider_table is None:
        return None
    keys = ["tournament", "match_id", "team"]
    cols = [c for c in PROVIDER_COLUMNS if c in provider_table]
    have = team_match[keys].merge(provider_table[keys + cols], on=keys, how="left")[cols].notna()
    have.index = team_match.index
    have = have.reindex(columns=list(PROVIDER_COLUMNS), fill_value=False)
    if excluded is not None and len(excluded):
        out = pd.MultiIndex.from_frame(team_match[["tournament", "match_id"]]).isin(
            pd.MultiIndex.from_frame(excluded[["tournament", "match_id"]]))
        have.loc[out, list(TRACKING_COLUMNS)] = False
    have["hsr_without_ball_per_dominance"] = have["hsr_without_ball_km"] & (team_match["dominance"] > 0)
    return have


def _coverage_check(rows: list, t: str, tm: pd.DataFrame, col: str, expected: pd.DataFrame | None) -> None:
    rule = COVERAGE[col]
    have = tm[col].notna()
    share = have.mean() if len(tm) else 0.0
    if rule == "all":
        _check(rows, f"{t}: {col} has a value", "100%", f"{100 * share:.1f}%", bool(have.all()))
    elif rule == "after_first":
        later = tm["match_number"] > 1
        _check(rows, f"{t}: {col} has a value (all but each team's first match)",
               f"{int(later.sum())} of {len(tm)}", f"{int(have.sum())} of {len(tm)}",
               bool((have == later).all()))
    elif rule == "defined":
        _check(rows, f"{t}: {col} has a value (empty only where the ratio is undefined)",
               "listed", f"{100 * share:.1f}% ({int((~have).sum())} empty)", None)
    elif t == "men" and expected is not None:
        want = expected.loc[tm.index, col]
        _check(rows, f"{t}: {col} has a value where team_match_provider has one (tracking columns empty for matches "
                     "over 30% filled in)", f"{int(want.sum())} of {len(tm)}", f"{int(have.sum())} of {len(tm)}",
               bool((have == want).all()))
    else:
        reason = "women: no tracking" if t != "men" else "team_match_provider not built yet"
        _check(rows, f"{t}: {col} has a value ({reason})", "empty", f"{100 * share:.1f}%",
               bool(not have.any()))


def _minutes_check(rows: list, t: str, pm: pd.DataFrame, lengths: pd.DataFrame) -> None:
    """Team minutes against 11 x match length; outliers must have a red card or permanent player off."""
    team = pm.groupby(["match_id", "team"]).agg(
        minutes=("minutes", "sum"),
        sent_off=("off_reason", lambda s: s.isin(["red card", "player off"]).any()))
    team = team.reset_index().merge(lengths[["match_id", "match_s"]], on="match_id", how="left")
    team["ratio"] = team["minutes"] / (STARTERS * team["match_s"] / 60)
    out = team[(team["ratio"] - 1).abs() > TEAM_MINUTES_TOLERANCE]
    listed = [f"{r.team} {r.match_id}: {r.ratio:.3f}" for r in out.itertuples()]
    _check(rows, f"{t}: team minutes within 1% of 11 x match length (outliers listed; each needs a red card "
                 "or permanent player off)", "all outliers explained",
           f"{len(team) - len(out)} of {len(team)} within; outliers {listed or 'none'}",
           bool(out["sent_off"].all()))


def check_round2(events: pd.DataFrame, team_match: pd.DataFrame, minutes: pd.DataFrame,
                 phase_match: pd.DataFrame, tournament_table: pd.DataFrame,
                 expected: pd.DataFrame | None) -> pd.DataFrame:
    """Checks of the context, load, chances, defensive, possession, spread and phase columns."""
    rows: list[dict] = []
    for key, text in DEFINITIONS.items():
        _check(rows, f"both: definition of {key}", "stated", text, None)
    poss_all = res.possessions(events)
    phase_poss = ph.possession_phases(events)
    lengths = res.period_lengths(events)
    credits = res.shot_credits(events)
    for t in sorted(team_match["tournament"].unique()):
        tm = team_match[team_match["tournament"] == t]
        pm = minutes[minutes["tournament"] == t]
        for col in NEW_COLUMNS:
            _coverage_check(rows, t, tm, col, expected)
        for col in UNIT_SHARE_COLUMNS:
            v = tm[col].dropna()
            _check(rows, f"{t}: {col} within 0-1", "all", f"{int(v.between(0, 1).sum())} of {len(v)}",
                   bool(v.between(0, 1).all()))
        sums = tm.groupby("match_id")["possession_share"].sum()
        ok = int(np.isclose(sums, 1.0).sum())
        _check(rows, f"{t}: possession_share of the two teams sums to 1", f"{len(sums)} of {len(sums)}",
               f"{ok} of {len(sums)}", ok == len(sums))
        over = int((tm["xa"] > tm["np_xg"] + 1e-9).sum())
        _check(rows, f"{t}: xa <= np_xg (assisted shots are a subset of the team's shots)", 0, over,
               over == 0)
        ct = credits[credits["tournament"] == t]
        assisted = ct[ct["passer_team_id"].notna()]
        cross = int((assisted["passer_team_id"] != assisted["team_id"]).sum())
        _check(rows, f"{t}: assisted shots whose key pass is by the other team", 0, cross, cross == 0)
        _check(rows, f"{t}: non-penalty shots with a key pass (xA credited)", "report",
               f"{len(assisted)} of {len(ct)}", None)
        for col in ("security", "conversion", "ppda", "defensive_work_per_dominance"):
            v = tm[col].dropna()
            finite = int(np.isfinite(v).sum())
            _check(rows, f"{t}: {col} finite where defined", "all", f"{finite} of {len(v)}",
                   bool(np.isfinite(v).all()))

        starters = pm[pm["started"]].groupby(["match_id", "team"]).size()
        ok = int((starters == STARTERS).sum())
        _check(rows, f"{t}: team-matches with 11 starters", f"{len(tm)} of {len(tm)}", f"{ok} of {len(tm)}",
               ok == len(tm))
        _minutes_check(rows, t, pm, lengths[lengths["tournament"] == t])
        no_group = int(pm["position_group"].isna().sum())
        _check(rows, f"{t}: players with a position group", f"{len(pm)} of {len(pm)}",
               f"{len(pm) - no_group} of {len(pm)}", no_group == 0)
        no_jersey = int(pm["jersey_number"].isna().sum())
        _check(rows, f"{t}: player rows with a jersey number (match lineups, else the team's other lineups "
                     "in the tournament; the rest appear in no lineup)",
               f"{len(pm)} of {len(pm)}", f"{len(pm) - no_jersey} of {len(pm)}", None)
        reasons = pm.loc[pm["off_reason"] != "", "off_reason"].value_counts().to_dict()
        _check(rows, f"{t}: player time ended by (substitution / red card / permanent player off)",
               "listed", reasons, None)
        most = tm.groupby("team")["match_number"].max()
        _check(rows, f"{t}: teams by number of matches played", "3 or more each",
               most.value_counts().sort_index().to_dict(), bool(most.min() >= 3))

        pt = poss_all[poss_all["tournament"] == t]
        phm = phase_match[phase_match["tournament"] == t]
        assigned = int(phm["possessions"].sum())
        _check(rows, f"{t}: possessions with a phase (play-event possessions, periods 1-2)",
               f"{len(pt)} of {len(pt)}", f"{assigned} of {len(pt)}", assigned == len(pt))
        per_tm = phm.groupby(["match_id", "team"])["possessions"].sum()
        _check(rows, f"{t}: possessions per team-match (min / median / max)", "listed",
               f"{per_tm.min()} / {per_tm.median():.0f} / {per_tm.max()}", None)
        mix = phm.groupby("phase")["possessions"].sum()
        _check(rows, f"{t}: possessions by phase (share)", "listed",
               {p: f"{100 * mix.get(p, 0) / max(mix.sum(), 1):.1f}%" for p in ph.PHASES}, None)
        pp = phase_poss[phase_poss["tournament"] == t]
        tr = pp[pp["phase"] == "transition"]
        flag = "is_controlled_possession"
        both = tr[flag].fillna(False).astype(bool).mean() if flag in tr else np.nan
        _check(rows, f"{t}: transition possessions also flagged controlled possession "
                     "(transition ranks above settled)", "listed", f"{100 * both:.1f}% of {len(tr)}", None)
        xg_gap = abs(phm["xg"].sum() - tm["np_xg"].sum())
        _check(rows, f"{t}: phase xG (shots in own possessions) against team non-penalty xG", "< 0.5 apart",
               f"{xg_gap:.3f}", xg_gap < 0.5)

        tt = tournament_table[tournament_table["tournament"] == t]
        target = TEAM_TOURNAMENT_ROWS.get(t)
        _check(rows, f"{t}: team_tournament rows", target, len(tt), len(tt) == target)
        for col in NEW_COLUMNS:
            if tm[col].notna().any():
                _check(rows, f"{t}: summary of {col} (team-match)", "report", _summary(tm[col]), None)
    _check(rows, "men: team_match_provider merged", "when built by the tracking stage",
           "merged" if expected is not None else "not built yet; men's tracking columns empty", None)
    _ireland_panama_round2(rows, team_match)
    return pd.DataFrame(rows, columns=["check", "target", "value", "passed"]).astype({"passed": "boolean"})


def _ireland_panama_round2(rows: list, team_match: pd.DataFrame) -> None:
    ire, pan = IRELAND_PANAMA
    tm = team_match[(team_match["tournament"] == "men") & team_match["team"].isin(IRELAND_PANAMA)]
    both = tm.groupby("match_id")["team"].nunique()
    if not both.eq(2).any():
        return
    m = tm[tm["match_id"] == both[both.eq(2)].index[0]].set_index("team")
    deck = {"possession_share": "possession %", "pressures": "pressures", "distance_km": "distance km"}
    for col in NEW_COLUMNS:
        a, b = m.loc[ire, col], m.loc[pan, col]
        if pd.isna(a) and pd.isna(b):
            continue
        scale = 100 if col == "possession_share" else 1
        source = "tracking / FIFA" if col in PROVIDER_COLUMNS else "StatsBomb"
        _check(rows, f"men: Ireland v Panama {col} ({source}; FIFA deck figure where one exists)",
               FIFA_DECK[deck[col]] if col in deck else "no FIFA figure",
               f"{scale * a:.3f} v {scale * b:.3f}", None)


def check_shot_timeline(timeline: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Goals in the shot timeline against each match score, both teams."""
    rows: list[dict] = []
    sides = dom.match_sides(matches)
    goals = timeline[timeline["goal"]].groupby(TEAM_KEYS).size().rename("timeline_goals")
    for t in sorted(matches["tournament"].unique()):
        m = matches[matches["tournament"] == t]
        home = m[["tournament", "match_id", "home_team_id", "home_score"]].set_axis(TEAM_KEYS + ["goals"], axis=1)
        away = m[["tournament", "match_id", "away_team_id", "away_score"]].set_axis(TEAM_KEYS + ["goals"], axis=1)
        both = pd.concat([home, away]).merge(goals.reset_index(), on=TEAM_KEYS, how="left").fillna(
            {"timeline_goals": 0})
        ok = int((both["goals"] == both["timeline_goals"]).sum())
        _check(rows, f"{t}: shot_timeline goals = match score (periods 1-2, own goals included)",
               f"{len(both)} of {len(both)}", f"{ok} of {len(both)}", ok == len(both))
        tl = timeline[timeline["tournament"] == t]
        n_sides = int(sides[sides["tournament"] == t].shape[0])
        _check(rows, f"{t}: shot_timeline shots and own goals", "listed",
               f"{int((~tl['own_goal']).sum())} shots, {int(tl['own_goal'].sum())} own goals, {n_sides} team-matches",
               None)
    return pd.DataFrame(rows, columns=["check", "target", "value", "passed"]).astype({"passed": "boolean"})


def check_excluded(excluded: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """The matches whose tracking columns are left empty, with their filled-in share."""
    from three_trees.tracking.running import EXCLUDE_FILLED_IN_SHARE

    named = excluded.merge(matches[["tournament", "match_id", "home_team", "away_team"]],
                           on=["tournament", "match_id"], how="left")
    listed = [f"{r.home_team} v {r.away_team} {100 * r.filled_in_share:.0f}%" for r in named.itertuples()]
    rows: list[dict] = []
    _check(rows, f"men: matches with more than {EXCLUDE_FILLED_IN_SHARE:.0%} of player positions filled in by the "
                 "provider (tracking columns empty)", "listed", listed or "none", None)
    return pd.DataFrame(rows, columns=["check", "target", "value", "passed"]).astype({"passed": "boolean"})


def run(cfg: Config) -> None:
    from three_trees.tracking.running import excluded_matches

    events = read_table(cfg, "events")
    matches = read_table(cfg, "matches")
    provider_table = read_table(cfg, "team_match_provider") if table_exists(cfg, "team_match_provider") else None
    running = read_table(cfg, "running_player_match") if table_exists(cfg, "running_player_match") else None
    excluded = excluded_matches(running)

    team_match, minutes = build_team_match(events, matches, provider_table, excluded)
    tournament_table = team_tournament(team_match)
    zone_match = dom.team_match_zone(events, matches)
    zone_tournament = dom.team_tournament_zone(zone_match)
    phase_match = ph.team_match_phase(events, matches)
    phase_tournament = ph.team_tournament_phase(phase_match)
    timeline = res.shot_timeline(events)
    listed = listed_formations(events)

    checks = pd.concat([
        check_features(events, matches, team_match, zone_match),
        check_round2(events, team_match, minutes, phase_match, tournament_table,
                     provider_expected(team_match, provider_table, excluded)),
        check_shot_timeline(timeline, matches),
        check_excluded(excluded, matches),
    ], ignore_index=True).astype({"passed": "boolean"})

    tables = {
        "team_match": team_match, "team_tournament": tournament_table,
        "team_match_zone": zone_match, "team_tournament_zone": zone_tournament,
        "team_match_phase": phase_match, "team_tournament_phase": phase_tournament,
        "player_match_minutes": minutes[res.PLAYER_MINUTES_COLUMNS],
        "shot_timeline": timeline,
        "formation_listed": listed,
    }
    for name, df in tables.items():
        write_table(cfg, name, df)
        print(f"  {name}: {len(df):,} rows, {df.shape[1]} columns")
    write_table(cfg, "checks_features", checks)
    with pd.option_context("display.width", 250, "display.max_colwidth", 160, "display.max_rows", 2000):
        print(checks.to_string(index=False))


