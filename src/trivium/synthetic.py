"""Synthetic demo build: a complete, clearly fake set of tables in the shape of the real ones,
for building and testing the dashboard and query box before the real data exists.

- Team names are "Team A1" ... ("Team <group letter><position in group>"), the same names in
  both tournaments (as real team names repeat across the men's and women's tournaments).
- Every value is random noise shaped to look plausible; nothing describes a real match or player.
- Table build_info marks the build as synthetic; the dashboard shows a banner from it.
- write_demo refuses a folder inside the repo, the configured real data root, or a build folder
  that already holds tables not marked synthetic.

Tables: matches, elo_pre_tournament, tds_team, team_match, team_match_zone,
team_tournament_zone, running_player_match, build_info, positions_window,
formation_team_window, formation_team_match and formation_listed; then the expectation stage on top.
"""

from __future__ import annotations

import contextlib
import io
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from trivium.config import Config, repo_root
from trivium.measures import measures
from trivium.storage import read_table, table_exists, write_table

SEED = 20251103
FORMATS = {"full": {"men": (12, 32), "women": (6, 16)}, "small": {"men": (6, 16), "women": (4, 8)}}
KNOCKOUT_LABELS = {32: "Round of 32", 16: "Round of 16", 8: "Quarter-finals", 4: "Semi-finals", 2: "Final"}
START = {"men": date(2025, 11, 3), "women": date(2025, 10, 17)}
ELO_MEAN_SD = {"men": (1800.0, 170.0), "women": (1950.0, 220.0)}
TEAM_ID_BASE = {"men": 1000, "women": 2000}
GROUP_FIXTURES = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)]
CONFEDERATIONS = ("AFC", "CAF", "Concacaf", "CONMEBOL", "OFC", "UEFA")
BEST_FINISHES = ("Champions", "Runners-up", "Third", "Fourth", "Quarter-finals", "Round of 16", "Group stage", None)
SQUAD = [("goalkeeper", "GK", "Goalkeeper")] + [("player", "DF", "Defender")] * 4 + \
    [("player", "MF", "Midfielder")] * 4 + [("player", "FW", "Forward")] * 2 + \
    [("player", "DF", "Defender"), ("player", "MF", "Midfielder"), ("player", "FW", "Forward")]
STATES = ("with_ball", "without_ball", "out_of_play")
THIRDS = ("defensive", "middle", "attacking")
ZONE_KEYS = ["tournament", "match_id", "team", "zone_x", "zone_y", "zone"]
# Stages run on the demo build after the expectation stage (the features stage is replaced by
# the synthetic team_match and zone tables).
MODEL_STAGES = ("reliability", "drivers", "styles", "profiles", "sustainability")
# FIFA Enhanced-file phases; the first FIFA_IN_POSSESSION are in possession.
FIFA_PHASES = ("build_up", "progression", "final_third", "long_ball", "attacking_transition", "counter_attack",
               "press", "counter_press", "defensive_transition", "recovery", "block")
FIFA_IN_POSSESSION = 6
STATSBOMB_PHASES = ("settled", "transition", "counter_attack", "set_piece", "other")
PROVIDER_COLUMNS = ("distance_km", "hsr_km", "sprint_km", "distance_km_tracking", "hsr_km_tracking", "hsr_with_ball_km",
               "hsr_without_ball_km", "hsr_out_of_play_km", "hsr_without_ball_defensive_third_km", "top3_hsr_share",
               "line_height_m", "team_length_m", "team_width_m", *(f"fifa_{p}_share" for p in FIFA_PHASES))
STARTING_POSITIONS = ("Goalkeeper", "Right Back", "Right Center Back", "Left Center Back", "Left Back",
                      "Right Defensive Midfield", "Left Defensive Midfield", "Right Wing", "Center Attacking Midfield",
                      "Left Wing", "Center Forward")
BENCH_POSITIONS = ("Center Back", "Center Midfield", "Right Wing", "Center Forward", "Left Back", "Left Wing", "Goalkeeper")
REPORT_PHASES_IN = ("build_up_unopposed", "build_up_opposed", "progression", "final_third", "long_ball",
                    "attacking_transition", "counter_attack", "set_piece")
REPORT_PHASES_OUT = ("high_press", "mid_press", "low_press", "high_block", "mid_block", "low_block", "recovery",
                     "defensive_transition", "counter_press")


def team_names(n_groups: int) -> list[str]:
    return [f"Team {chr(ord('A') + g)}{i}" for g in range(n_groups) for i in range(1, 5)]


def demo_matches(
    tournament: str, n_groups: int, first_round: int, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(matches, elo_pre_tournament) for one synthetic tournament: groups of four, best thirds,
    single-elimination knockout with a third-place match; the first team in group A is host."""
    teams = team_names(n_groups)
    mean, sd = ELO_MEAN_SD[tournament]
    elo = dict(zip(teams, rng.normal(mean, sd, len(teams))))
    host = teams[0]
    start = START[tournament]
    rows: list[dict] = []

    def play(home: str, away: str, stage: str, day: int, knockout: bool) -> str:
        gap = (elo[home] - elo[away]) / 400
        hs = int(rng.poisson(np.exp(0.2 + 0.55 * gap)))
        as_ = int(rng.poisson(np.exp(0.2 - 0.55 * gap)))
        winner = None
        if knockout and hs == as_:
            winner = home if rng.random() < 1 / (1 + np.exp(-gap)) else away
        rows.append(
            {
                "match_id": len(rows) + 1,
                "match_date": start + timedelta(days=day),
                "competition_stage": stage,
                "home_team": home,
                "away_team": away,
                "home_score": hs,
                "away_score": as_,
                "neutral_ground": home != host and away != host,
                "shootout": winner is not None,
                "shootout_winner": winner,
            }
        )
        return winner or (home if hs > as_ else away)

    points = dict.fromkeys(teams, 0)
    goal_diff = dict.fromkeys(teams, 0)
    winners, runners, thirds = [], [], []
    for g in range(n_groups):
        group = teams[4 * g: 4 * g + 4]
        for k, (i, j) in enumerate(GROUP_FIXTURES):
            home, away = (group[i], group[j]) if group[j] != host else (group[j], group[i])
            play(home, away, "Group Stage", 4 * (k // 2) + g % 2, False)
            hs, as_ = rows[-1]["home_score"], rows[-1]["away_score"]
            points[home] += 3 if hs > as_ else 1 if hs == as_ else 0
            points[away] += 3 if as_ > hs else 1 if hs == as_ else 0
            goal_diff[home] += hs - as_
            goal_diff[away] += as_ - hs
        ranked = sorted(group, key=lambda t: (points[t], goal_diff[t], elo[t]), reverse=True)
        winners.append(ranked[0])
        runners.append(ranked[1])
        thirds.append(ranked[2])
    best_thirds = sorted(thirds, key=lambda t: (points[t], goal_diff[t], elo[t]), reverse=True)
    alive = winners + runners + best_thirds[: first_round - 2 * n_groups]

    day = 12
    while len(alive) > 1:
        pairs = [(alive[i], alive[-1 - i]) for i in range(len(alive) // 2)]
        next_round = [play(a, b, KNOCKOUT_LABELS[len(alive)], day, True) for a, b in pairs]
        if len(alive) == 4:
            losers = [b if w == a else a for (a, b), w in zip(pairs, next_round)]
            play(losers[0], losers[1], "3rd Place Final", day + 3, True)
        alive = next_round
        day += 4

    ids = {t: TEAM_ID_BASE[tournament] + i for i, t in enumerate(teams)}
    matches = pd.DataFrame(rows).assign(tournament=tournament, kick_off="18:00:00.000")
    matches["home_team_id"] = matches["home_team"].map(ids)
    matches["away_team_id"] = matches["away_team"].map(ids)
    matches = matches[
        ["tournament", "match_id", "match_date", "kick_off", "competition_stage", "home_team", "away_team",
         "home_team_id", "away_team_id", "home_score", "away_score", "neutral_ground", "shootout", "shootout_winner"]
    ]
    elo_pre = pd.DataFrame(
        {
            "tournament": tournament,
            "team": teams,
            "elo_name": teams,
            "elo": [elo[t] for t in teams],
            "senior_matches": rng.integers(150, 900, len(teams)),
            "last_senior_match": start - timedelta(days=20),
        }
    )
    elo_pre["elo_rank_in_tournament"] = elo_pre["elo"].rank(ascending=False, method="min").astype("Int64")
    return matches, elo_pre.sort_values("elo_rank_in_tournament", ignore_index=True)


def demo_tds(teams: list[str], rng: np.random.Generator) -> pd.DataFrame:
    finishes = rng.choice(len(BEST_FINISHES), len(teams))
    campaigns = rng.integers(1, 20, len(teams))
    best = [BEST_FINISHES[i] for i in finishes]
    return pd.DataFrame(
        {
            "team": teams,
            "elo_name": teams,
            "confederation": rng.choice(CONFEDERATIONS, len(teams)),
            "campaigns_incl_2025": campaigns,
            "last_qualified": pd.array([None if b is None else 2023 for b in best], dtype="Int64"),
            "best_finish_before_2025": best,
            "best_finish_rank": [8 if b is None else i + 1 for i, b in zip(finishes, best)],
        }
    )


def _sides(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match with the match's latent strength gap and score."""
    sides = []
    for side, other in (("home", "away"), ("away", "home")):
        sides.append(
            matches[["tournament", "match_id", "match_date"]].assign(
                team=matches[f"{side}_team"],
                opponent=matches[f"{other}_team"],
                goals_for=matches[f"{side}_score"],
                goals_against=matches[f"{other}_score"],
                is_home=side == "home",
            )
        )
    return pd.concat(sides, ignore_index=True)


def demo_team_match(matches: pd.DataFrame, elo_pre: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Every team_match measure in the registry, mirrored across the two sides of each match."""
    m = matches.reset_index(drop=True)
    ratings = elo_pre.set_index(["tournament", "team"])["elo"]
    gap = np.array([ratings[(t, h)] - ratings[(t, a)] for t, h, a in
                    zip(m["tournament"], m["home_team"], m["away_team"])]) / 400
    n = len(m)
    tilt = np.clip(0.5 + 0.18 * np.tanh(1.5 * gap) + rng.normal(0, 0.07, n), 0.08, 0.92)
    contests = np.clip(0.5 + 0.06 * np.tanh(gap) + rng.normal(0, 0.04, n), 0.3, 0.7)
    goal_gap = (m["home_score"] - m["away_score"]).to_numpy()
    chances = np.clip(0.5 + 0.2 * np.tanh(1.5 * gap) + 0.06 * goal_gap + rng.normal(0, 0.08, n), 0.05, 0.95)
    possession = np.clip(0.5 + 0.15 * np.tanh(1.5 * gap) + rng.normal(0, 0.06, n), 0.2, 0.8)
    xg_total = np.maximum(0.6, rng.normal(2.5, 0.7, n))

    home = {"territory_share": tilt, "contests_share": contests, "chances_share": chances,
            "possession_share": possession}
    for part in ("duels_share", "aerials_share", "fifty_fifties_share", "second_balls_share"):
        home[part] = np.clip(contests + rng.normal(0, 0.07, n), 0.1, 0.9)
    rows = _sides(m)
    is_home = rows["is_home"].to_numpy()
    idx = np.tile(np.arange(n), 2)
    for col, values in home.items():
        rows[col] = np.where(is_home, values[idx], 1 - values[idx])
    final_third_total = np.maximum(20, rng.normal(110, 25, n))
    rows["final_third_actions"] = np.round(rows["territory_share"] * final_third_total[idx]).astype(int)
    aerial_skill = rng.normal(0, 1.2, n)
    rows["aerials_won_above_hops"] = np.where(is_home, aerial_skill[idx], -aerial_skill[idx])
    rows["np_xg"] = rows["chances_share"] * xg_total[idx]
    rows["np_xg_against"] = (1 - rows["chances_share"]) * xg_total[idx]
    rows["dominance"] = rows[["contests_share", "territory_share", "chances_share"]].mean(axis=1)
    rows["conversion"] = rows["np_xg"] / rows["territory_share"]
    k = len(rows)
    rows["xa"] = rows["np_xg"] * rng.uniform(0.55, 0.85, k)
    entries_against = np.maximum(5, 15 + 50 * rows["territory_share"].rsub(1) + rng.normal(0, 5, k))
    rows["opponent_final_third_entries"] = np.round(entries_against).astype(int)
    rows["security"] = rows["np_xg_against"] / rows["opponent_final_third_entries"]
    def_actions = np.round(120 + 80 * (1 - rows["possession_share"]) + rng.normal(0, 15, k))
    rows["defensive_actions"] = def_actions.astype(int)
    rows["pressures"] = np.round(0.6 * def_actions).astype(int)
    rows["defensive_work_per_dominance"] = rows["defensive_actions"] / (rows["dominance"] * 100)
    rows["possession_minutes"] = 55 * rows["possession_share"]
    rows["pressures_per_opp_possession_min"] = rows["pressures"] / (55 * (1 - rows["possession_share"]))
    rows["ppda"] = np.clip(14 - 12 * (rows["possession_share"] - 0.5) + rng.normal(0, 2, k), 3, 30)
    rows["directness"] = np.clip(0.45 - 0.2 * (rows["possession_share"] - 0.5) + rng.normal(0, 0.06, k), 0.1, 0.9)
    rows["top2_xg_xa_share"] = np.clip(rng.normal(0.45, 0.12, k), 0.15, 0.95)
    rows["chance_zone_spread"] = np.clip(rng.normal(0.6, 0.1, k), 0.1, 1.0)
    rows["players_used"] = 11 + rng.integers(2, 6, k)

    men = (rows["tournament"] == "men").to_numpy()
    distance = rng.normal(105, 4, k)
    hsr = rng.normal(8.5, 1.2, k)
    hsr_tracking = hsr * rng.uniform(1.02, 1.08, k)
    without_share = np.clip(0.5 - 0.3 * (rows["possession_share"] - 0.5) + rng.normal(0, 0.05, k), 0.2, 0.7)
    without_ball = hsr_tracking * without_share
    with_ball = hsr_tracking * (0.85 - without_share)
    provider_table = {
        "distance_km": distance,
        "hsr_km": hsr,
        "sprint_km": hsr * rng.uniform(0.25, 0.35, k),
        "distance_km_tracking": distance * rng.uniform(1.0, 1.012, k),
        "hsr_km_tracking": hsr_tracking,
        "hsr_with_ball_km": with_ball,
        "hsr_without_ball_km": without_ball,
        "hsr_out_of_play_km": hsr_tracking * 0.15,
        "hsr_without_ball_defensive_third_km": without_ball * rng.uniform(0.35, 0.5, k),
        "hsr_without_ball_per_dominance": without_ball * 1000 / (rows["dominance"] * 100),
        "top3_hsr_share": np.clip(rng.normal(0.3, 0.04, k), 0.18, 0.5),
        "line_height_m": np.clip(38 + 24 * (rows["territory_share"] - 0.5) + rng.normal(0, 3, k), 22, 55),
        "team_length_m": np.clip(rng.normal(36, 3, k), 25, 50),
        "team_width_m": np.clip(rng.normal(42, 3, k), 30, 55),
    }
    possession = rows["possession_share"].to_numpy()[:, None]
    alpha = np.where(np.arange(len(FIFA_PHASES)) < FIFA_IN_POSSESSION, 8 * possession, 8 * (1 - possession))
    shares = rng.gamma(alpha + 0.5)
    shares /= shares.sum(axis=1, keepdims=True)
    for i, phase in enumerate(FIFA_PHASES):
        provider_table[f"fifa_{phase}_share"] = shares[:, i]
    for col, values in provider_table.items():
        rows[col] = np.where(men, values, np.nan)

    rows = rows.sort_values(["tournament", "team", "match_date", "match_id"], ignore_index=True)
    g = rows.groupby(["tournament", "team"], sort=False)
    rows["match_number"] = g.cumcount() + 1
    dates = pd.to_datetime(rows["match_date"])
    rows["rest_days"] = dates.groupby([rows["tournament"], rows["team"]], sort=False).diff().dt.days
    rows["starters_changed"] = np.where(rows["match_number"] == 1, np.nan,
                                        np.minimum(rng.poisson(2.0, len(rows)), 11)).astype(float)
    rows["minutes_outside_top11_share"] = np.clip(
        0.02 * rows["match_number"] + rng.beta(2, 18, len(rows)), 0, 0.6
    )
    played = rows.groupby(["tournament", "team"], sort=False)["match_number"].transform(lambda s: s - 1)
    rows["starters_minutes_cum"] = played * 990 * rng.uniform(0.75, 0.95, len(rows))

    columns = measures(table="team_match")["column"].tolist()
    return rows[["tournament", "match_id", "team", "opponent", *columns]].sort_values(
        ["tournament", "match_id", "team"], ignore_index=True
    )


def demo_zones(team_match: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """team_match_zone (18 zones, 6 x 3, from the measured team's view) and team_tournament_zone."""
    zx, zy = np.meshgrid(np.arange(6), np.arange(3), indexing="ij")
    zx, zy = zx.ravel(), zy.ravel()
    parts = []
    for row in team_match[["tournament", "match_id", "team", "territory_share", "contests_share"]].itertuples(
        index=False
    ):
        lift = 2 * (row.territory_share - 0.5)
        team_rate = 22 * np.exp(0.25 * (zx - 2.5) * (1 + lift)) * (1.3 - 0.3 * np.abs(zy - 1))
        opp_rate = 22 * np.exp(0.25 * (2.5 - zx) * (1 - lift)) * (1.3 - 0.3 * np.abs(zy - 1))
        team_actions = rng.poisson(team_rate)
        opponent_actions = rng.poisson(opp_rate)
        contests = rng.poisson(4.0 * (1.4 - 0.15 * np.abs(zx - 2.5)))
        won = rng.binomial(contests, row.contests_share)
        parts.append(
            pd.DataFrame(
                {
                    "tournament": row.tournament,
                    "match_id": row.match_id,
                    "team": row.team,
                    "zone_x": zx,
                    "zone_y": zy,
                    "zone": zx * 3 + zy + 1,
                    "contests_won": won,
                    "contests_lost": contests - won,
                    "team_actions": team_actions,
                    "opponent_actions": opponent_actions,
                }
            )
        )
    zone = pd.concat(parts, ignore_index=True)
    zone = _zone_shares(zone)
    totals = zone.groupby(["tournament", "team", "zone_x", "zone_y", "zone"], as_index=False)[
        ["contests_won", "contests_lost", "team_actions", "opponent_actions"]
    ].sum()
    return zone, _zone_shares(totals)


def _zone_shares(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    contests = df["contests_won"] + df["contests_lost"]
    df["contests_share"] = np.where(contests > 0, df["contests_won"] / contests.where(contests > 0, 1), 0.5)
    df["actions_count"] = df["team_actions"] + df["opponent_actions"]
    df["territory_share"] = np.where(
        df["actions_count"] > 0, df["team_actions"] / df["actions_count"].where(df["actions_count"] > 0, 1), 0.5
    )
    return df


def demo_running(team_match: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """running_player_match for the men: one row per synthetic player x possession state x third.
    Players carry ids, shirt numbers and positions only; there are no names."""
    men = team_match[team_match["tournament"] == "men"]
    teams = sorted(men["team"].unique())
    squad = len(SQUAD)
    parts = []
    for row in men[["tournament", "match_id", "team", "possession_share", "hsr_km"]].itertuples(index=False):
        base = 100_000 + 100 * teams.index(row.team)
        starters = np.r_[np.ones(11), np.zeros(squad - 11)]
        minutes = np.where(starters == 1, rng.uniform(60, 96, squad), rng.uniform(0, 30, squad))
        minutes[0] = 96.0
        share = {"with_ball": row.possession_share * 0.45, "without_ball": (1 - row.possession_share) * 0.45}
        share["out_of_play"] = 1 - share["with_ball"] - share["without_ball"]
        for state in STATES:
            for k, third in enumerate(THIRDS):
                third_weight = np.array([0.35, 0.4, 0.25]) if state != "with_ball" else np.array([0.25, 0.4, 0.35])
                frac = share[state] * third_weight[k]
                distance = minutes * 115 * frac * rng.uniform(0.85, 1.15, squad)
                hsr = distance * np.where([p[1] == "GK" for p in SQUAD], 0.01, 0.08) * rng.uniform(0.7, 1.3, squad)
                parts.append(
                    pd.DataFrame(
                        {
                            "tournament": row.tournament,
                            "match_id": row.match_id,
                            "team": row.team,
                            "person_id": base + np.arange(squad),
                            "jersey_number": np.arange(1, squad + 1),
                            "category_name": [p[0] for p in SQUAD],
                            "position": [p[1] for p in SQUAD],
                            "position_group": [p[2] for p in SQUAD],
                            "state": state,
                            "third": third,
                            "minutes": minutes * frac,
                            "distance_m": distance,
                            "hsr_m": hsr,
                            "sprint_m": hsr * rng.uniform(0.2, 0.35, squad),
                        }
                    )
                )
    running = pd.concat(parts, ignore_index=True)
    return running[running["minutes"] > 0].reset_index(drop=True)


def _with_tracking_columns(running: pd.DataFrame) -> pd.DataFrame:
    """The tracking stage's extra running columns: tracking team name, frame counts, FIFA speed bands."""
    out = running.copy()
    out["tracking_team"] = out["team"].str.upper()
    out["frames"] = np.round(out["minutes"] * 60 * 25).astype(int)
    out["imputed_frames"] = np.round(out["frames"] * 0.09).astype(int)
    out["speed_missing_frames"] = 0
    out["dist_1_7_kmh_m"] = (out["distance_m"] - out["hsr_m"]) * 0.4
    out["dist_7_15_kmh_m"] = (out["distance_m"] - out["hsr_m"]) * 0.45
    out["dist_15_20_kmh_m"] = (out["distance_m"] - out["hsr_m"]) * 0.15
    out["dist_20_25_kmh_m"] = out["hsr_m"] - out["sprint_m"]
    out["dist_25_plus_kmh_m"] = out["sprint_m"]
    return out


def _position_group(position: str) -> str:
    if position == "Goalkeeper":
        return "Goalkeeper"
    if "Back" in position:
        return "Defender"
    return "Midfielder" if "Midfield" in position else "Forward"


def demo_player_minutes(team_match: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """player_match_minutes: 11 starters and the substitutes per team-match (ids and shirt numbers only)."""
    index = {key: i for i, key in enumerate(sorted(set(zip(team_match["tournament"], team_match["team"]))))}
    parts = []
    for r in team_match[["tournament", "match_id", "team", "players_used"]].itertuples(index=False):
        base = 500_000 + 100 * index[(r.tournament, r.team)]
        subs = int(r.players_used) - 11
        replaced = rng.choice(np.arange(1, 11), size=subs, replace=False)
        minute = np.sort(rng.uniform(55, 85, subs))
        minutes, off = np.full(11, 95.0), np.full(11, np.nan)
        minutes[replaced], off[replaced] = minute, minute
        starters = pd.DataFrame({"player_id": base + np.arange(11), "jersey_number": np.arange(1, 12),
                                 "position": STARTING_POSITIONS, "started": True, "minutes": minutes,
                                 "on_minute": 0.0, "off_minute": off})
        bench = pd.DataFrame({"player_id": base + 11 + np.arange(subs), "jersey_number": 12 + np.arange(subs),
                              "position": BENCH_POSITIONS[:subs], "started": False, "minutes": 95 - minute,
                              "on_minute": minute, "off_minute": np.nan})
        parts.append(pd.concat([starters, bench], ignore_index=True).assign(
            tournament=r.tournament, match_id=r.match_id, team=r.team))
    df = pd.concat(parts, ignore_index=True)
    df["position_group"] = df["position"].map(_position_group)
    return df[["tournament", "match_id", "team", "player_id", "jersey_number", "position", "position_group",
               "started", "minutes", "on_minute", "off_minute"]]


def demo_phases(team_match: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """team_match_phase and team_tournament_phase (StatsBomb phases of play)."""
    rows = []
    for r in team_match[["tournament", "match_id", "team", "np_xg", "possession_minutes"]].itertuples(index=False):
        share = rng.dirichlet(np.array([0.45, 0.2, 0.05, 0.2, 0.1]) * 40)
        possessions = np.maximum(1, np.round(rng.normal(85, 10) * share)).astype(int)
        xg = r.np_xg * rng.dirichlet(np.array([0.4, 0.25, 0.1, 0.2, 0.05]) * 30)
        shots = rng.poisson(np.maximum(xg * 9, 0.1))
        seconds = r.possession_minutes * 60 * share
        for i, phase in enumerate(STATSBOMB_PHASES):
            rows.append({"tournament": r.tournament, "match_id": r.match_id, "team": r.team, "phase": phase,
                         "possessions": int(possessions[i]), "possession_seconds": seconds[i], "shots": int(shots[i]),
                         "xg": xg[i], "obv": float(rng.normal(0.05, 0.1) * share[i] * 10),
                         "xg_per_possession": xg[i] / possessions[i],
                         "share_of_possessions": possessions[i] / possessions.sum()})
    match_phase = pd.DataFrame(rows)
    sums = ["possessions", "possession_seconds", "shots", "xg", "obv"]
    totals = match_phase.groupby(["tournament", "team", "phase"], as_index=False)[sums].sum()
    totals["xg_per_possession"] = totals["xg"] / totals["possessions"]
    totals["share_of_possessions"] = totals["possessions"] / totals.groupby(["tournament", "team"])[
        "possessions"].transform("sum")
    return match_phase, totals


def demo_shot_timeline(matches: pd.DataFrame, team_match: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """shot_timeline: shots over 95 minutes with the team's np_xg split between them; goals = the score."""
    xg = team_match.set_index(["tournament", "match_id", "team"])["np_xg"]
    rows = []
    for m in matches.itertuples(index=False):
        for team, team_id, goals in ((m.home_team, m.home_team_id, m.home_score), (m.away_team, m.away_team_id,
                                                                                  m.away_score)):
            n = max(int(goals), int(rng.poisson(11)))
            elapsed = np.sort(rng.uniform(0.5, 95, n))
            shot_xg = xg[(m.tournament, m.match_id, team)] * rng.dirichlet(np.ones(n) * 0.8)
            scored = rng.choice(n, int(goals), replace=False, p=shot_xg / shot_xg.sum()) if goals else []
            for i, minute in enumerate(elapsed):
                period = 1 if minute < 47 else 2
                clock = minute if period == 1 else minute - 2
                rows.append({"tournament": m.tournament, "match_id": m.match_id, "team": team, "team_id": team_id,
                             "period": period, "minute": int(clock), "second": int(60 * (clock % 1)),
                             "elapsed_minute": float(minute), "half_time_minute": 47.0, "xg": float(shot_xg[i]),
                             "goal": i in scored, "penalty": False, "own_goal": False})
    return pd.DataFrame(rows)


def demo_team_tournament(team_match: pd.DataFrame) -> pd.DataFrame:
    """team_tournament: matches and the per-match mean of every numeric team_match measure."""
    numeric = [c for c in team_match.columns
               if c != "match_id" and pd.api.types.is_numeric_dtype(team_match[c])]
    return team_match.groupby(["tournament", "team"], as_index=False).agg(
        matches=("match_id", "nunique"), **{c: (c, "mean") for c in numeric})


def demo_provider_tables(team_match: pd.DataFrame, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """Men's match-folder tables: team_match_provider, phase_clips, running_fifa_phase, shape_team_second,
    event_running_windows, player_lookup and fifa_events (team or position-group level, no names)."""
    keys = ["tournament", "match_id", "team"]
    men = team_match[team_match["tournament"] == "men"].reset_index(drop=True)
    clips, running, shape, windows, fifa = [], [], [], [], []
    groups = (("Goalkeeper", 1, 0.9, 0.005), ("Defender", 4, 1.9, 0.06), ("Midfielder", 4, 2.1, 0.08),
              ("Forward", 2, 2.0, 0.1))
    for r in men.itertuples(index=False):
        key = {"tournament": r.tournament, "match_id": r.match_id, "team": r.team}
        for phase in FIFA_PHASES:
            share = getattr(r, f"fifa_{phase}_share")
            starts = rng.uniform(0, 2600, 3)
            clips += [{**key, "phase": phase, "period": int(rng.integers(1, 3)), "start_s": s,
                       "end_s": s + rng.uniform(5, 40)} for s in starts]
            seconds = share * 3000
            for group, players, speed, hsr_rate in groups:
                distance = seconds * speed * players * rng.uniform(0.9, 1.1)
                hsr = distance * hsr_rate * rng.uniform(0.8, 1.2)
                running.append({**key, "position_group": group, "phase": phase, "seconds": seconds,
                                "distance_m": distance, "hsr_m": hsr, "sprint_m": hsr * 0.3})
        n = 90
        with_ball = rng.random(n) < r.possession_share
        shape.append(pd.DataFrame({**key, "period": np.repeat([1, 2], n // 2), "clock_s": np.tile(np.arange(45) * 60.0, 2),
                                   "state": np.where(with_ball, "with_ball", "without_ball"),
                                   "line_height_m": r.line_height_m + rng.normal(0, 4, n) + 8 * with_ball,
                                   "team_length_m": r.team_length_m + rng.normal(0, 3, n),
                                   "team_width_m": r.team_width_m + rng.normal(0, 3, n)}))
        m = int(r.defensive_actions)
        windows.append(pd.DataFrame({**key, "event_id": [f"syn-{r.match_id}-{r.team[-2:]}-{i}" for i in range(m)],
                                     "type": rng.choice(["Pressure", "Duel", "Interception", "Ball Recovery"], m,
                                                        p=[0.6, 0.15, 0.1, 0.15]),
                                     "location_x": rng.uniform(5, 115, m), "location_y": rng.uniform(0, 80, m),
                                     "team_hsr_10s_m": rng.gamma(2.0, 20.0, m),
                                     "team_distance_10s_m": rng.normal(320, 40, m)}))
        e = 120
        fifa.append(pd.DataFrame({**key,
                                  "category": rng.choice(["in_possession", "out_of_possession", "in_contest"], e),
                                  "event_type": rng.choice(["pass", "carry", "pressure", "shot", "duel"], e),
                                  "event": rng.choice(["pass", "cross", "pressure", "tackle", "shot"], e),
                                  "outcome": rng.choice(["complete", "incomplete", "won", "lost"], e),
                                  "game_period": rng.integers(1, 3, e),
                                  "match_run_time_in_ms": np.sort(rng.integers(0, 5_700_000, e)),
                                  "pressure": rng.choice(["no", "indirect", "direct"], e),
                                  "line_break_outcome": rng.choice(["", "complete", "incomplete"], e),
                                  "line_break_direction": rng.choice(["", "through", "around", "over"], e),
                                  "team_units_broken": rng.integers(0, 3, e), "total_team_units": 3,
                                  "team_shape": rng.choice(["inside", "outside"], e),
                                  "team_unit": rng.choice(["in_front", "between_1_2", "between_2_3", "in_behind"], e),
                                  "sequence_type": rng.choice(["open_play", "set_piece"], e),
                                  "x_location_start": rng.uniform(0, 105, e), "y_location_start": rng.uniform(0, 68, e),
                                  "x_location_end": rng.uniform(0, 105, e), "y_location_end": rng.uniform(0, 68, e)}))
    return {
        "team_match_provider": men[[*keys, *PROVIDER_COLUMNS]],
        "phase_clips": pd.DataFrame(clips),
        "running_fifa_phase": pd.DataFrame(running),
        "shape_team_second": pd.concat(shape, ignore_index=True),
        "event_running_windows": pd.concat(windows, ignore_index=True),
        "fifa_events": pd.concat(fifa, ignore_index=True),
    }


def demo_player_lookup(player_minutes: pd.DataFrame) -> pd.DataFrame:
    """player_lookup (men): StatsBomb, FIFA and tracking ids matched on team and shirt number; synthetic
    tracking ids follow demo_running, which tracks shirt numbers 1-14."""
    men = player_minutes[player_minutes["tournament"] == "men"].copy()
    teams = {t: i for i, t in enumerate(sorted(men["team"].unique()))}
    idx = men["team"].map(teams)
    tracked = men["jersey_number"] <= len(SQUAD)
    men["statsbomb_player_id"] = men["player_id"]
    men["fifa_player_id"] = 900_000 + 100 * idx + men["jersey_number"]
    men["tracking_person_id"] = (100_000 + 100 * idx + men["jersey_number"] - 1).where(tracked).astype("Int64")
    men["method"] = np.where(tracked, "shirt number", "unmatched")
    return men[["tournament", "match_id", "team", "statsbomb_player_id", "fifa_player_id", "tracking_person_id",
                "jersey_number", "position_group", "method"]].reset_index(drop=True)


def demo_reports(matches: pd.DataFrame, team_match: pd.DataFrame,
                 rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """report_team_match (FIFA post-match report figures) and provider_check_team_match."""
    goals = pd.concat([
        matches[["tournament", "match_id", "home_team", "home_score"]].set_axis(["tournament", "match_id", "team", "goals"], axis=1),
        matches[["tournament", "match_id", "away_team", "away_score"]].set_axis(["tournament", "match_id", "team", "goals"], axis=1),
    ], ignore_index=True)
    tm = team_match.merge(goals, on=["tournament", "match_id", "team"], how="left", validate="one_to_one")
    k = len(tm)
    in_contest = np.clip(rng.normal(10, 3, k), 3, 25)
    passes = np.round(np.clip(350 + 300 * (tm["possession_share"] - 0.5) + rng.normal(0, 40, k), 120, 800))
    complete = np.round(passes * rng.uniform(0.72, 0.9, k))
    attempts = rng.poisson(tm["np_xg"] * 9 + 3)
    report = tm[["tournament", "match_id", "team", "goals"]].assign(
        possession_pct=tm["possession_share"] * (100 - in_contest), possession_in_contest_pct=in_contest,
        attempts=attempts, attempts_on_target=rng.binomial(attempts, 0.35), passes=passes.astype(int),
        passes_complete=complete.astype(int), pass_completion_pct=100 * complete / passes,
        line_breaks_completed=rng.poisson(80, k), defensive_line_breaks=rng.poisson(80, k),
        receptions_final_third=rng.poisson(40, k), crosses=rng.poisson(12, k), ball_progressions=rng.poisson(30, k),
        pressures_applied=np.round(tm["pressures"] * rng.uniform(1.3, 1.7, k)).astype(int),
        direct_pressures=np.round(tm["pressures"] * rng.uniform(0.6, 0.9, k)).astype(int),
        forced_turnovers=rng.poisson(12, k), second_balls=rng.poisson(20, k),
        distance_km=tm["distance_km"], zone4_km=tm["hsr_km"] * 0.7)
    for names in (REPORT_PHASES_IN, REPORT_PHASES_OUT):
        shares = rng.dirichlet(np.ones(len(names)) * 4, k) * 100
        for i, name in enumerate(names):
            report[f"phase_{name}_pct"] = shares[:, i]
    pairs = (
        ("possession", tm["possession_share"] * 100, report["possession_pct"], "pass share of the match (%)",
         "possession time excluding in-contest (%)"),
        ("shots", attempts * rng.uniform(0.85, 1.15, k), report["attempts"], "shots", "attempts at goal"),
        ("passes", passes * rng.uniform(0.9, 1.1, k), report["passes"], "passes", "total passes"),
        ("pressures", tm["pressures"], report["pressures_applied"], "pressure events", "defensive pressures applied"),
        ("distance km", tm["distance_km_tracking"], report["distance_km"], "tracking distance", "FIFA physical distance"),
    )
    check = pd.concat([
        tm[["tournament", "match_id", "team"]].assign(measure=m, statsbomb_value=np.asarray(s, dtype=float),
                                                       fifa_value=np.asarray(f, dtype=float),
                                                       statsbomb_definition=sd, fifa_definition=fd)
        for m, s, f, sd, fd in pairs
    ], ignore_index=True)
    return report, check.dropna(subset=["statsbomb_value", "fifa_value"]).reset_index(drop=True)


def demo_positions_window(matches: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Make nameless, synthetic player positions arranged in recognisable tactical lines."""
    rng = np.random.default_rng(seed + 47_113)
    shapes = ("4-4-2", "4-3-3", "3-5-2", "4-2-3-1", "5-3-2")
    teams = sorted(set(matches.loc[matches["tournament"].eq("men"), "home_team"])
                   | set(matches.loc[matches["tournament"].eq("men"), "away_team"]))
    team_index = {team: i for i, team in enumerate(teams)}
    rows = []
    men_matches = matches.loc[matches["tournament"].eq("men")]
    for match in men_matches.itertuples(index=False):
        for team in (match.home_team, match.away_team):
            index = team_index[team]
            base_shape = shapes[index % len(shapes)]
            for period in (1, 2):
                for window in range(9):
                    switched = (match.match_id + period * 3 + window + index) % 13 == 0
                    shape = shapes[(index + 1) % len(shapes)] if switched else base_shape
                    lines = [int(value) for value in shape.split("-")]
                    line_depths = np.linspace(25, 70, len(lines))
                    for state, shift in (("with_ball", 5.0), ("without_ball", -4.0)):
                        person = 0
                        for line, count in enumerate(lines):
                            widths = np.linspace(10, 58, count) if count > 1 else np.array([34.0])
                            group = "Defender" if line == 0 else "Forward" if line == len(lines) - 1 else "Midfielder"
                            for width in widths:
                                rows.append({
                                    "tournament": "men",
                                    "match_id": match.match_id,
                                    "team": team,
                                    "period": period,
                                    "window": window,
                                    "window_start_min": (period - 1) * 45 + window * 5,
                                    "state": state,
                                    "person_id": 300_000 + index * 100 + person,
                                    "position_group": group,
                                    "x_m": float(line_depths[line] + shift + rng.normal(0, 0.25)),
                                    "y_m": float(width + rng.normal(0, 1.2)),
                                    "seconds": 300.0,
                                })
                                person += 1
    return pd.DataFrame(rows, columns=[
        "tournament", "match_id", "team", "period", "window", "window_start_min", "state", "person_id",
        "position_group", "x_m", "y_m", "seconds",
    ])


def demo_listed_formations(matches: pd.DataFrame) -> pd.DataFrame:
    """Make fake listed shapes for both tournaments, with occasional tactical changes."""
    shapes = ("4-4-2", "4-3-3", "3-5-2", "4-2-3-1", "5-3-2")
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    team_index = {team: i for i, team in enumerate(teams)}
    rows = []
    for match in matches.itertuples(index=False):
        for team in (match.home_team, match.away_team):
            index = team_index[team] + (1 if match.tournament == "women" else 0)
            starting = shapes[index % len(shapes)]
            changed = (match.match_id + index) % 7 == 0
            main = shapes[(index + 1) % len(shapes)] if changed else starting
            rows.append({
                "tournament": match.tournament,
                "match_id": match.match_id,
                "team": team,
                "starting_formation": starting,
                "main_formation": main,
                "formation_changes": int(changed),
                "back_line": f"back {main.split('-')[0]}",
            })
    return pd.DataFrame(rows, columns=[
        "tournament", "match_id", "team", "starting_formation", "main_formation", "formation_changes", "back_line",
    ])


def demo_tables(seed: int = SEED, scale: str = "full") -> dict[str, pd.DataFrame]:
    """Every synthetic input table, for both tournaments."""
    rng = np.random.default_rng(seed)
    match_parts, elo_parts = [], []
    for tournament, (n_groups, first_round) in FORMATS[scale].items():
        matches_t, elo_t = demo_matches(tournament, n_groups, first_round, rng)
        match_parts.append(matches_t)
        elo_parts.append(elo_t)
    matches = pd.concat(match_parts, ignore_index=True)
    elo_pre = pd.concat(elo_parts, ignore_index=True)
    team_match = demo_team_match(matches, elo_pre, rng)
    zone, zone_totals = demo_zones(team_match, rng)
    men_teams = sorted(elo_pre.loc[elo_pre["tournament"] == "men", "team"])
    player_minutes = demo_player_minutes(team_match, rng)
    match_phase, tournament_phase = demo_phases(team_match, rng)
    report, provider_check = demo_reports(matches, team_match, rng)
    tables = {
        "matches": matches,
        "elo_pre_tournament": elo_pre,
        "tds_team": demo_tds(men_teams, rng),
        "team_match": team_match,
        "team_tournament": demo_team_tournament(team_match),
        "team_match_zone": zone,
        "team_tournament_zone": zone_totals,
        "running_player_match": _with_tracking_columns(demo_running(team_match, rng)),
        "player_match_minutes": player_minutes,
        "player_lookup": demo_player_lookup(player_minutes),
        "team_match_phase": match_phase,
        "team_tournament_phase": tournament_phase,
        **demo_provider_tables(team_match, rng),
        "report_team_match": report,
        "provider_check_team_match": provider_check,
        "shot_timeline": demo_shot_timeline(matches, team_match, rng),
        "build_info": pd.DataFrame(
            {"source": ["synthetic"], "seed": [seed], "scale": [scale],
             "note": ["Synthetic demo data: random values, fake teams; not real matches."]}
        ),
    }
    from trivium.tracking.formations import team_match_formations, team_window_formations

    positions = demo_positions_window(matches, seed)
    window_formations = team_window_formations(positions)
    tables.update({
        "positions_window": positions,
        "formation_team_window": window_formations,
        "formation_team_match": team_match_formations(window_formations),
        "formation_listed": demo_listed_formations(matches),
    })
    return tables


def demo_config(out: Path) -> Config:
    out = Path(out).expanduser().resolve()
    return Config(
        source=out / "synthetic",
        data_root=out,
        build_dir=out / "build",
        statsbomb={},
        match_folders_men=out / "match_folders",
        fifa_reports_women=out / "fifa_reports_women",
        external_dir=out / "external",
        external_urls={},
    )


def is_synthetic(cfg: Config) -> bool:
    return table_exists(cfg, "build_info") and read_table(cfg, "build_info")["source"].eq("synthetic").all()


def write_demo(
    out: Path,
    seed: int = SEED,
    scale: str = "full",
    run_expectation: bool = True,
    luck_sims: int = 2_000,
    progress_sims: int = 2_000,
    real_data_root: Path | None = None,
    run_models: bool = False,
) -> Config:
    """Write the synthetic build to <out>/build and, by default, run the expectation stage on it;
    with run_models, also every model stage after it and the validation report."""
    cfg = demo_config(out)
    root = repo_root()
    if root is not None and cfg.data_root.is_relative_to(root):
        raise ValueError(f"Demo data must be written outside the repo ({root}).")
    if real_data_root is not None and cfg.data_root == Path(real_data_root).expanduser().resolve():
        raise ValueError(f"{cfg.data_root} is the real data root; choose a separate folder for demo data.")
    if cfg.build_dir.is_dir() and any(cfg.build_dir.glob("*.parquet")) and not is_synthetic(cfg):
        raise ValueError(f"{cfg.build_dir} holds tables that are not synthetic; choose an empty folder.")
    for name, df in demo_tables(seed, scale).items():
        write_table(cfg, name, df)
    if run_expectation:
        from trivium.models import expectation

        with contextlib.redirect_stdout(io.StringIO()):
            expectation.run(cfg, luck_sims=luck_sims, progress_sims=progress_sims)
    if run_expectation and run_models:
        import importlib

        from trivium.cli import STAGES, VALIDATION

        after = [module for name, module in STAGES if name in MODEL_STAGES]
        with contextlib.redirect_stdout(io.StringIO()):
            for module in [*after, VALIDATION]:
                importlib.import_module(module).run(cfg)
    return cfg
