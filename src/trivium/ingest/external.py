"""Public external data: senior Elo the day before kick-off, computed from the
martj42 international results (men before 2025-11-03, women before
2025-10-17), and the FIFA TDS youth record from lookup/tds_record.csv.
Team names are mapped with lookup/countries.csv.

Output tables: elo_pre_tournament, tds_team, checks_external.
"""

from __future__ import annotations

import shutil
import unicodedata
import urllib.request
from collections.abc import Callable
from datetime import date
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from trivium.config import TOURNAMENTS, Config
from trivium.storage import read_table, write_table
from trivium.teams import elo_name

RESULTS_KEYS = {"men": "results_men", "women": "results_women"}
CUTOFFS = {"men": date(2025, 11, 3), "women": date(2025, 10, 17)}
START_RATING = 1500.0
HOME_ADVANTAGE = 100.0

# World Football Elo K factors (build context section 6). Names are compared
# case- and accent-insensitively.
K_QUALIFICATION = 40
K_WORLD_CUP = 60
K_OLYMPICS = 50
K_CONTINENTAL = 50
K_NATIONS_LEAGUE = 40
K_FRIENDLY = 20
K_OTHER = 30
WORLD_CUP = frozenset({"fifa world cup"})
OLYMPICS = frozenset({"olympic games"})
CONTINENTAL_FINALS = frozenset(
    {
        "uefa euro",
        "copa america",
        "african cup of nations",
        "africa cup of nations",
        "afc asian cup",
        "gold cup",
        "concacaf championship",
        "oceania nations cup",
        "ofc nations cup",
        "confederations cup",
    }
)

# Senior Elo in the planning run (build context section 6), for the checks.
ELO_TARGETS = {
    "men": {
        "Argentina": 2185, "France": 2112, "England": 2088, "Portugal": 2078, "Colombia": 2069,
        "Tajikistan": 1434, "Fiji": 1458, "Indonesia": 1489,
    },
    "women": {
        "United States": 2374, "Spain": 2357, "France": 2280, "Japan": 2245, "Brazil": 2207,
        "Ecuador": 1442, "Samoa": 1502, "Costa Rica": 1640,
    },
}
ELO_TOLERANCE = 10.0

BEST_FINISH_RANK = {
    "Champions": 1,
    "Runners-up": 2,
    "Third": 3,
    "Fourth": 4,
    "Quarter-finals": 5,
    "Round of 16": 6,
    "Group stage": 7,
}
NO_PREVIOUS_CAMPAIGN = 8


# ---------------------------------------------------------------- results


def results_path(cfg: Config, tournament: str) -> Path:
    return cfg.external_dir / f"{RESULTS_KEYS[tournament]}.csv"


def download_results(cfg: Config, force: bool = False) -> dict[str, Path]:
    """Download the martj42 results for both tournaments; existing files are kept unless force."""
    cfg.external_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for tournament in TOURNAMENTS:
        path = results_path(cfg, tournament)
        if force or not path.is_file():
            tmp = path.with_suffix(".csv.tmp")
            with urllib.request.urlopen(cfg.external_urls[RESULTS_KEYS[tournament]], timeout=60) as response:
                with tmp.open("wb") as f:
                    shutil.copyfileobj(response, f)
            tmp.replace(path)
        paths[tournament] = path
    return paths


def load_results(path: Path) -> pd.DataFrame:
    """Results with parsed dates and booleans; rows without a score (fixtures) dropped."""
    df = pd.read_csv(path, dtype={"home_team": str, "away_team": str, "tournament": str})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df[["home_score", "away_score"]] = df[["home_score", "away_score"]].astype(int)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- Elo


def _fold(name: str) -> str:
    text = unicodedata.normalize("NFKD", name.casefold())
    return "".join(c for c in text if not unicodedata.combining(c)).strip()


def k_factor(tournament: str) -> int:
    t = _fold(tournament)
    if "qualification" in t:
        return K_QUALIFICATION
    if t in WORLD_CUP:
        return K_WORLD_CUP
    if t in OLYMPICS:
        return K_OLYMPICS
    if t in CONTINENTAL_FINALS:
        return K_CONTINENTAL
    if "nations league" in t:
        return K_NATIONS_LEAGUE
    if t == "friendly":
        return K_FRIENDLY
    return K_OTHER


def goal_index(goal_difference: int) -> float:
    """G: 1 for a margin of 0 or 1, 1.5 for 2, (11 + N) / 8 for N >= 3."""
    n = abs(goal_difference)
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    return (11 + n) / 8


def expected_score(rating_difference: float | np.ndarray) -> float | np.ndarray:
    """We = 1 / (10^(-dr/400) + 1)."""
    return 1.0 / (10.0 ** (-np.asarray(rating_difference) / 400.0) + 1.0)


def compute_elo(
    results: pd.DataFrame, cutoff: date, k: Callable[[str], int] = k_factor
) -> pd.DataFrame:
    """Ratings after every match dated before the cutoff, processed in date order."""
    played = results[results["date"] < cutoff].sort_values("date", kind="stable")
    k_by_tournament = {name: k(name) for name in played["tournament"].unique()}
    ratings: dict[str, float] = {}
    matches: dict[str, int] = {}
    last: dict[str, date] = {}
    for day, home, away, hs, as_, tournament, neutral in zip(
        played["date"], played["home_team"], played["away_team"], played["home_score"],
        played["away_score"], played["tournament"], played["neutral"],
    ):
        rh = ratings.get(home, START_RATING)
        ra = ratings.get(away, START_RATING)
        dr = rh - ra + (0.0 if neutral else HOME_ADVANTAGE)
        w = 1.0 if hs > as_ else 0.5 if hs == as_ else 0.0
        change = k_by_tournament[tournament] * goal_index(hs - as_) * (w - float(expected_score(dr)))
        ratings[home] = rh + change
        ratings[away] = ra - change
        for team in (home, away):
            matches[team] = matches.get(team, 0) + 1
            last[team] = day
    return (
        pd.DataFrame(
            {
                "elo_name": list(ratings),
                "elo": list(ratings.values()),
                "senior_matches": [matches[t] for t in ratings],
                "last_senior_match": [last[t] for t in ratings],
            }
        )
        .sort_values("elo", ascending=False, ignore_index=True)
    )


def elo_pre_tournament(elo: pd.DataFrame, teams: pd.Series, tournament: str) -> pd.DataFrame:
    """Elo of each tournament team (clean StatsBomb name); unmatched teams have no elo."""
    df = pd.DataFrame({"team": sorted(set(teams))})
    df.insert(0, "tournament", tournament)
    df["elo_name"] = df["team"].map(elo_name)
    df = df.merge(elo, on="elo_name", how="left")
    df["elo_rank_in_tournament"] = df["elo"].rank(ascending=False, method="min").astype("Int64")
    return df.sort_values("elo_rank_in_tournament", ignore_index=True)


def tournament_teams(matches: pd.DataFrame, tournament: str) -> pd.Series:
    m = matches[matches["tournament"] == tournament]
    return pd.concat([m["home_team"], m["away_team"]], ignore_index=True)


# ---------------------------------------------------------------- TDS record


def load_tds() -> pd.DataFrame:
    with (files("trivium") / "lookup" / "tds_record.csv").open(encoding="utf-8", newline="") as f:
        df = pd.read_csv(f, dtype={"best_finish_before_2025": str})
    df["last_qualified"] = df["last_qualified"].astype("Int64")
    df["best_finish_rank"] = (
        df["best_finish_before_2025"].map(BEST_FINISH_RANK).fillna(NO_PREVIOUS_CAMPAIGN).astype(int)
    )
    return df


def tds_team(tds: pd.DataFrame, teams: pd.Series) -> pd.DataFrame:
    """TDS record per men's tournament team (clean StatsBomb name); unmatched teams have no record."""
    df = pd.DataFrame({"team": sorted(set(teams))})
    df["elo_name"] = df["team"].map(elo_name)
    return df.merge(tds.rename(columns={"team": "elo_name"}), on="elo_name", how="left")


# ---------------------------------------------------------------- checks


def baseline_check(matches: pd.DataFrame, ratings: pd.DataFrame, stage: str = "group") -> dict[str, float]:
    """Test 2 on matches whose stage contains `stage`: does the higher-rated team win?

    ratings: team, rating (any strength measure where higher is stronger).
    Brier uses the Elo expected score with +100 for the host (neutral_ground false).
    """
    m = matches[matches["competition_stage"].str.contains(stage, case=False, na=False)]
    r = ratings.set_index("team")["rating"]
    home, away = m["home_team"].map(r), m["away_team"].map(r)
    w = np.select([m["home_score"] > m["away_score"], m["home_score"] == m["away_score"]], [1.0, 0.5], 0.0)
    decisive = (w != 0.5) & (home != away) & home.notna() & away.notna()
    higher_won = ((home > away) & (w == 1.0)) | ((home < away) & (w == 0.0))
    dr = home - away + np.where(m["neutral_ground"].astype(bool), 0.0, HOME_ADVANTAGE)
    we = expected_score(dr.to_numpy(dtype=float))
    return {
        "matches": float(len(m)),
        "decisive": float(decisive.sum()),
        "higher_rated_won": float((higher_won & decisive).sum()),
        "higher_rated_share": float((higher_won & decisive).sum() / decisive.sum()) if decisive.any() else np.nan,
        "brier": float(np.nanmean((we - w) ** 2)),
        "brier_coin": float(np.mean((0.5 - w) ** 2)),
    }


def elo_target_checks(elo: pd.DataFrame, tournament: str) -> pd.DataFrame:
    by_name = elo.set_index("elo_name")["elo"]
    rows = []
    for team, target in ELO_TARGETS[tournament].items():
        value = by_name.get(team, np.nan)
        rows.append(
            {
                "check": f"{tournament} Elo {team}",
                "target": float(target),
                "value": round(float(value), 1),
                "passed": bool(abs(value - target) <= ELO_TOLERANCE),
            }
        )
    return pd.DataFrame(rows)


def baseline_checks(matches: pd.DataFrame, ratings: pd.DataFrame, label: str, targets: dict[str, float]) -> list[dict]:
    """Test 2 rows: higher-rated share of decisive group matches (binomial test
    against 0.5), and Brier against the coin baseline."""
    b = baseline_check(matches, ratings)
    p_value = binomtest(int(b["higher_rated_won"]), int(b["decisive"]), 0.5, alternative="greater").pvalue
    rows = [
        _check(f"{label}: higher-rated team wins, share of {int(b['decisive'])} decisive group matches",
               targets.get("share", np.nan), round(b["higher_rated_share"], 3), p_value < 0.05),
        _check(f"{label}: binomial p-value (higher-rated wins > 0.5)", 0.05, round(p_value, 4), p_value < 0.05),
    ]
    if "brier" in targets:
        rows += [
            _check(f"{label}: Brier, group matches", targets["brier"], round(b["brier"], 3),
                   b["brier"] < b["brier_coin"]),
            _check(f"{label}: Brier coin baseline", targets["brier_coin"], round(b["brier_coin"], 3), True),
        ]
    return rows


def _check(name: str, target: float, value: float, passed: bool) -> dict[str, object]:
    return {"check": name, "target": float(target), "value": float(value), "passed": bool(passed)}


# Planning-run values for test 2 (build context section 6).
BASELINE_TARGETS = {
    "men": {"share": 0.707, "brier": 0.158, "brier_coin": 0.201},
    "women": {"share": 0.788, "brier": 0.136, "brier_coin": 0.229},
    "men_tds_best_finish": {"share": 0.694},
    "men_tds_campaigns": {"share": 0.577},
}


# ---------------------------------------------------------------- stage


def run(cfg: Config) -> None:
    matches = read_table(cfg, "matches")
    paths = download_results(cfg)

    elo_tables, checks = [], []
    for tournament in TOURNAMENTS:
        elo = compute_elo(load_results(paths[tournament]), CUTOFFS[tournament])
        pre = elo_pre_tournament(elo, tournament_teams(matches, tournament), tournament)
        elo_tables.append(pre)
        checks.append(elo_target_checks(elo, tournament))
        unmatched = pre.loc[pre["elo"].isna(), "team"].tolist()
        rows = [_check(f"{tournament} teams matched to Elo (unmatched: {', '.join(unmatched) or 'none'})",
                       len(pre), len(pre) - len(unmatched), not unmatched)]
        t_matches = matches[matches["tournament"] == tournament]
        rows += baseline_checks(t_matches, pre.rename(columns={"elo": "rating"}), f"{tournament} Elo",
                                BASELINE_TARGETS[tournament])
        checks.append(pd.DataFrame(rows))

    elo_pre = pd.concat(elo_tables, ignore_index=True)
    tds_men = tds_team(load_tds(), tournament_teams(matches, "men"))
    men = elo_pre[elo_pre["tournament"] == "men"].merge(tds_men[["team", "campaigns_incl_2025"]], on="team")
    men_matches = matches[matches["tournament"] == "men"]
    rows = [
        _check("men teams matched to TDS record", tds_men["team"].nunique(),
               tds_men["confederation"].notna().sum(), tds_men["confederation"].notna().all()),
        _check("men corr(Elo, TDS campaigns)", 0.51, round(men["elo"].corr(men["campaigns_incl_2025"]), 3), True),
    ]
    rows += baseline_checks(men_matches, tds_men.assign(rating=-tds_men["best_finish_rank"]),
                            "men TDS best finish", BASELINE_TARGETS["men_tds_best_finish"])
    rows += baseline_checks(men_matches, tds_men.assign(rating=tds_men["campaigns_incl_2025"]),
                            "men TDS campaigns", BASELINE_TARGETS["men_tds_campaigns"])
    checks.append(pd.DataFrame(rows))
    checks_df = pd.concat(checks, ignore_index=True)

    write_table(cfg, "elo_pre_tournament", elo_pre)
    write_table(cfg, "tds_team", tds_men)
    write_table(cfg, "checks_external", checks_df)
    with pd.option_context("display.max_rows", None, "display.width", 160, "display.max_colwidth", 90):
        print(checks_df.to_string(index=False))
