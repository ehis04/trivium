from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from three_trees.config import DATA_ROOT_ENV, load_config
from three_trees.ingest import external as ex

EXAMPLE = Path(__file__).resolve().parents[1] / "config.example.yaml"
RESULTS_HEADER = "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    return load_config(EXAMPLE)


def _results(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"]
    )


# ---------------------------------------------------------------- download and load


def test_download_keeps_existing_files(cfg, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg.external_dir.mkdir(parents=True)
    for t in ("men", "women"):
        ex.results_path(cfg, t).write_text(RESULTS_HEADER, encoding="utf-8")

    def no_network(*args, **kwargs):
        raise AssertionError("download attempted")

    monkeypatch.setattr(ex.urllib.request, "urlopen", no_network)
    paths = ex.download_results(cfg)
    assert set(paths) == {"men", "women"}


def test_download_force_fetches(cfg, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ex.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(RESULTS_HEADER.encode()))
    paths = ex.download_results(cfg, force=True)
    assert paths["men"].read_text(encoding="utf-8") == RESULTS_HEADER
    assert not list(cfg.external_dir.glob("*.tmp"))


def test_load_results_drops_fixtures_and_parses(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    path.write_text(
        RESULTS_HEADER
        + "2020-01-01,A,B,2,1,Friendly,x,y,FALSE\n"
        + "2020-02-01,B,A,1,1,Friendly,x,y,TRUE\n"
        + "2030-01-01,A,B,NA,NA,Friendly,x,y,FALSE\n",
        encoding="utf-8",
    )
    df = ex.load_results(path)
    assert len(df) == 2
    assert df["neutral"].tolist() == [False, True]
    assert df["date"].iloc[0] == date(2020, 1, 1)
    assert df["home_score"].dtype.kind == "i"


# ---------------------------------------------------------------- Elo rules


@pytest.mark.parametrize(
    "tournament, k",
    [
        ("Friendly", 20),
        ("FIFA World Cup", 60),
        ("FIFA World Cup qualification", 40),
        ("UEFA Euro", 50),
        ("UEFA Euro qualification", 40),
        ("Copa América", 50),
        ("African Cup of Nations", 50),
        ("Africa Cup of Nations", 50),
        ("AFC Asian Cup", 50),
        ("Gold Cup", 50),
        ("CONCACAF Championship", 50),
        ("Oceania Nations Cup", 50),
        ("OFC Nations Cup", 50),
        ("Confederations Cup", 50),
        ("Olympic Games", 50),
        ("UEFA Nations League", 40),
        ("CONCACAF Nations League qualification", 40),
        ("Algarve Cup", 30),
        ("AFC Championship", 30),
        ("AFC Olympic Qualifying Tournament", 30),
    ],
)
def test_k_factor(tournament: str, k: int) -> None:
    assert ex.k_factor(tournament) == k


@pytest.mark.parametrize("gd, g", [(0, 1.0), (1, 1.0), (-1, 1.0), (2, 1.5), (-2, 1.5), (3, 14 / 8), (5, 2.0)])
def test_goal_index(gd: int, g: float) -> None:
    assert ex.goal_index(gd) == pytest.approx(g)


def test_expected_score() -> None:
    assert ex.expected_score(0) == pytest.approx(0.5)
    assert ex.expected_score(400) == pytest.approx(1 / 1.1)
    assert ex.expected_score(-400) == pytest.approx(1 - 1 / 1.1)


def test_compute_elo_by_hand() -> None:
    results = _results(
        [
            (date(2020, 1, 1), "A", "B", 2, 0, "Friendly", False),  # A home: dr = +100, K 20, G 1.5
            (date(2020, 6, 1), "B", "A", 1, 0, "FIFA World Cup", True),  # neutral, K 60, G 1
        ]
    )
    elo = ex.compute_elo(results, date(2021, 1, 1)).set_index("elo_name")
    assert elo.loc["A", "elo"] == pytest.approx(1478.935691, abs=1e-5)
    assert elo.loc["B", "elo"] == pytest.approx(1521.064309, abs=1e-5)
    assert elo["elo"].mean() == pytest.approx(ex.START_RATING)
    assert elo.loc["A", "senior_matches"] == 2
    assert elo.loc["A", "last_senior_match"] == date(2020, 6, 1)


def test_compute_elo_excludes_cutoff_day() -> None:
    results = _results(
        [
            (date(2025, 11, 2), "A", "B", 1, 0, "Friendly", True),
            (date(2025, 11, 3), "B", "A", 5, 0, "Friendly", True),
        ]
    )
    elo = ex.compute_elo(results, date(2025, 11, 3)).set_index("elo_name")
    assert elo.loc["A", "senior_matches"] == 1
    assert elo.loc["A", "elo"] > elo.loc["B", "elo"]


def test_elo_target_checks_flag_differences() -> None:
    elo = pd.DataFrame({"elo_name": ["Argentina", "France"], "elo": [2185.4, 2130.0]})
    checks = ex.elo_target_checks(elo, "men").set_index("check")
    assert checks.loc["men Elo Argentina", "passed"]
    assert not checks.loc["men Elo France", "passed"]
    assert not checks.loc["men Elo Fiji", "passed"]


# ---------------------------------------------------------------- tournament teams and TDS


def test_elo_pre_tournament_maps_aliases_and_lists_unmatched() -> None:
    elo = pd.DataFrame({"elo_name": ["Italy", "South Korea"], "elo": [1800.0, 1700.0]})
    pre = ex.elo_pre_tournament(elo, pd.Series(["Korea Republic", "Italy", "Italy", "Atlantis"]), "men")
    by_team = pre.set_index("team")
    assert len(pre) == 3
    assert by_team.loc["Korea Republic", "elo"] == 1700.0
    assert by_team.loc["Italy", "elo_rank_in_tournament"] == 1
    assert np.isnan(by_team.loc["Atlantis", "elo"])


def test_load_tds() -> None:
    tds = ex.load_tds().set_index("team")
    assert len(tds) == 48
    assert tds.loc["Brazil", "best_finish_rank"] == 1
    assert tds.loc["Mali", "best_finish_rank"] == 2
    assert tds.loc["Uganda", "best_finish_rank"] == ex.NO_PREVIOUS_CAMPAIGN
    assert pd.isna(tds.loc["Republic of Ireland", "last_qualified"])


def test_tds_team_uses_elo_names() -> None:
    out = ex.tds_team(ex.load_tds(), pd.Series(["Korea Republic", "Qatar"])).set_index("team")
    assert out.loc["Korea Republic", "campaigns_incl_2025"] == 8
    assert out.loc["Qatar", "confederation"] == "AFC"


# ---------------------------------------------------------------- baseline check


def _matches() -> pd.DataFrame:
    rows = [
        (1, "Group Stage", "A", "B", 2, 0, True),  # higher wins
        (2, "Group Stage", "C", "D", 0, 1, True),  # lower wins
        (3, "Group Stage", "A", "C", 1, 1, True),  # draw
        (4, "Group Stage", "B", "D", 3, 0, False),  # higher wins, B is host
        (5, "Round of 16", "A", "D", 0, 1, True),  # not group stage
    ]
    return pd.DataFrame(
        rows,
        columns=["match_id", "competition_stage", "home_team", "away_team", "home_score", "away_score",
                 "neutral_ground"],
    )


def test_baseline_check() -> None:
    ratings = pd.DataFrame({"team": ["A", "B", "C", "D"], "rating": [2000.0, 1800.0, 1600.0, 1400.0]})
    b = ex.baseline_check(_matches(), ratings)
    assert b["matches"] == 4
    assert b["decisive"] == 3
    assert b["higher_rated_won"] == 2
    assert b["higher_rated_share"] == pytest.approx(2 / 3)
    assert b["brier_coin"] == pytest.approx(0.1875)
    we = ex.expected_score(np.array([200.0, 200.0, 400.0, 400.0 + ex.HOME_ADVANTAGE]))
    w = np.array([1.0, 0.0, 0.5, 1.0])
    assert b["brier"] == pytest.approx(np.mean((we - w) ** 2))


def test_baseline_check_skips_equal_ratings() -> None:
    ratings = pd.DataFrame({"team": ["A", "B", "C", "D"], "rating": [1.0, 1.0, 1.0, 2.0]})
    b = ex.baseline_check(_matches(), ratings)
    assert b["decisive"] == 2
    assert b["higher_rated_won"] == 1
