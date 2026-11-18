"""Senior check (test 10: does the method hold in the professional game?).

Part 1, the expectation method on senior FIFA World Cups from the public martj42 results:
Elo the day before each tournament, the leave-team-out ordered logit, luck bands and the Elo
baseline, edition by edition, against base rates. Editions with a 16-team knockout: men
1998-2022 (32 teams), women 2015 and 2019 (24 teams with best thirds, the U17 women's format)
and 2023 (32 teams). Senior scores include extra time (the U17 data has none); shootout winners
come from the martj42 shootouts file.

Part 2, signifiers 1, 2, 3 and 6 on StatsBomb senior open data (models/senior_signifiers.py),
on the senior_events and senior_matches tables of the open-data adapter (ingest/open_data.py),
which runs first when those tables are not built.

Output tables: senior_backtest, checks_senior; Excel senior_backtest.xlsx.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from three_trees.config import TOURNAMENTS, Config
from three_trees.ingest import external as ex
from three_trees.models import expectation as xp
from three_trees.models import uncertainty
from three_trees.storage import table_exists, write_table

EDITIONS = {"men": (1998, 2002, 2006, 2010, 2014, 2018, 2022), "women": (2015, 2019, 2023)}
KNOCKOUT_LABELS = ["Round of 16"] * 8 + ["Quarter-finals"] * 4 + ["Semi-finals"] * 2 + ["3rd Place Final", "Final"]
GROUP_MATCHES = (36, 48)
LUCK_SIMS = 10_000
SEED = 20251103


def shootouts_path(cfg: Config, tournament: str) -> Path:
    return cfg.external_dir / f"shootouts_{tournament}.csv"


def download_shootouts(cfg: Config, force: bool = False) -> dict[str, Path]:
    """The martj42 shootouts file beside each results file; existing files are kept unless force."""
    cfg.external_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for tournament in TOURNAMENTS:
        path = shootouts_path(cfg, tournament)
        if force or not path.is_file():
            url = cfg.external_urls[ex.RESULTS_KEYS[tournament]].rsplit("/", 1)[0] + "/shootouts.csv"
            tmp = path.with_suffix(".csv.tmp")
            with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as f:
                shutil.copyfileobj(response, f)
            tmp.replace(path)
        paths[tournament] = path
    return paths


def load_shootouts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df[["date", "home_team", "away_team", "winner"]]


def senior_world_cup(results: pd.DataFrame, shootouts: pd.DataFrame, tournament: str,
                     year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(matches, elo_pre) for one senior World Cup, in the columns of the real tables."""
    wc = results[(results["tournament"] == "FIFA World Cup")
                 & (pd.to_datetime(results["date"]).dt.year == year)].sort_values("date", kind="stable")
    n_group = len(wc) - len(KNOCKOUT_LABELS)
    if n_group not in GROUP_MATCHES:
        raise ValueError(f"{tournament} {year}: {len(wc)} World Cup matches, not a 16-team knockout format.")
    wc = wc.reset_index(drop=True).merge(shootouts, on=["date", "home_team", "away_team"], how="left")
    matches = pd.DataFrame({
        "tournament": tournament,
        "match_id": range(1, len(wc) + 1),
        "match_date": wc["date"],
        "kick_off": "",
        "competition_stage": ["Group Stage"] * n_group + KNOCKOUT_LABELS,
        "home_team": wc["home_team"],
        "away_team": wc["away_team"],
        "home_team_id": 0,
        "away_team_id": 0,
        "home_score": wc["home_score"].astype(int),
        "away_score": wc["away_score"].astype(int),
        "neutral_ground": wc["neutral"].astype(bool),
        "shootout": wc["winner"].notna(),
        "shootout_winner": wc["winner"].fillna(""),
    })
    elo = ex.compute_elo(results, matches["match_date"].min())
    elo_pre = ex.elo_pre_tournament(elo, ex.tournament_teams(matches, tournament), tournament)
    missing = elo_pre.loc[elo_pre["elo"].isna(), "team"].tolist()
    if missing:
        raise ValueError(f"{tournament} {year}: no senior matches before the tournament for {', '.join(missing)}.")
    return matches, elo_pre


def backtest_edition(matches: pd.DataFrame, elo_pre: pd.DataFrame, luck_sims: int = LUCK_SIMS,
                     seed: int = SEED) -> dict[str, float]:
    """Leave-team-out model vs base rates, calibration, luck bands and the Elo baseline for one edition."""
    rows = xp.team_match_rows(matches, elo_pre).reset_index(drop=True)
    probs = xp.leave_team_out(rows, xp.ordered_logit())
    base = xp.leave_team_out(rows, xp.base_rates).to_numpy()
    model_scores = xp.score_predictions(rows["result"], probs.to_numpy())
    base_scores = xp.score_predictions(rows["result"], base)
    table = xp.overperformance(rows, probs)
    bands = uncertainty.luck_band(table, n_sims=luck_sims, seed=seed)
    luck = uncertainty.luck_summary(bands).iloc[0]
    calibration = uncertainty.calibration_error(uncertainty.calibration_table(table)).iloc[0]
    baseline = ex.baseline_check(matches, elo_pre.rename(columns={"elo": "rating"}))
    return {
        "teams": float(rows["team"].nunique()),
        "matches": float(matches["match_id"].nunique()),
        "rps_model": model_scores["rps"],
        "rps_base_rates": base_scores["rps"],
        "log_loss_model": model_scores["log_loss"],
        "log_loss_base_rates": base_scores["log_loss"],
        "model_beats_base_rates": float(model_scores["rps"] < base_scores["rps"]),
        "win_probability_calibration_error": float(calibration["win_probability_error"]),
        "points_calibration_error": float(calibration["points_error"]),
        "teams_beyond_luck": float(luck["above"] + luck["below"]),
        "beyond_luck_expected_by_chance": float(luck["expected_by_chance"]),
        "higher_rated_share_group": baseline["higher_rated_share"],
        "brier_group": baseline["brier"],
        "brier_coin_group": baseline["brier_coin"],
        "draw_share": float((rows["result"] == xp.DRAW).mean()),
    }


def backtest(results: dict[str, pd.DataFrame], shootouts: dict[str, pd.DataFrame],
             editions: dict[str, tuple[int, ...]] = EDITIONS, luck_sims: int = LUCK_SIMS) -> pd.DataFrame:
    rows = []
    for tournament, years in editions.items():
        for year in years:
            matches, elo_pre = senior_world_cup(results[tournament], shootouts[tournament], tournament, year)
            rows.append({"tournament": tournament, "year": year, **backtest_edition(matches, elo_pre, luck_sims)})
    return pd.DataFrame(rows)


def _check(name: str, target: float, value: float, passed: bool) -> dict[str, object]:
    return {"check": name, "target": float(target), "value": float(value), "passed": bool(passed)}


def checks_senior(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tournament, t in table.groupby("tournament", sort=False):
        n, better = len(t), int(t["model_beats_base_rates"].sum())
        rows += [
            _check(f"senior {tournament}: editions where the model beats base rates (out-of-sample)", n, better,
                   better > n / 2),
            _check(f"senior {tournament}: mean ranked probability score, model vs base rates", float(
                t["rps_base_rates"].mean()), float(t["rps_model"].mean()),
                t["rps_model"].mean() < t["rps_base_rates"].mean()),
            _check(f"senior {tournament}: teams beyond luck, all editions (expected by chance as target)",
                   float(t["beyond_luck_expected_by_chance"].sum()), float(t["teams_beyond_luck"].sum()), True),
            _check(f"senior {tournament}: higher-rated team wins, share of decisive group matches", 0.5,
                   float(t["higher_rated_share_group"].mean()), t["higher_rated_share_group"].mean() > 0.5),
        ]
    return pd.DataFrame(rows)


def run(cfg: Config) -> None:
    results_paths = ex.download_results(cfg)
    shootout_paths = download_shootouts(cfg)
    results = {t: ex.load_results(results_paths[t]) for t in TOURNAMENTS}
    shootouts = {t: load_shootouts(shootout_paths[t]) for t in TOURNAMENTS}
    table = backtest(results, shootouts)
    checks = checks_senior(table)
    write_table(cfg, "senior_backtest", table)
    write_table(cfg, "checks_senior", checks)
    from three_trees.ingest import open_data
    from three_trees.models.senior_signifiers import run_senior_signifiers

    if not (table_exists(cfg, "senior_events") and table_exists(cfg, "senior_matches")):
        open_data.run(cfg)
    run_senior_signifiers(cfg, results=results)
    xp.write_excel(cfg.build_dir / "senior_backtest.xlsx", {"senior_backtest": table, "checks_senior": checks})
    cols = ["tournament", "year", "teams", "rps_model", "rps_base_rates", "points_calibration_error",
            "teams_beyond_luck", "higher_rated_share_group", "draw_share"]
    with pd.option_context("display.width", 160):
        print(table[cols].round(3).to_string(index=False))
        print(checks.to_string(index=False))
    if not np.isfinite(table["rps_model"]).all():
        raise ValueError("Senior backtest produced non-finite scores.")
