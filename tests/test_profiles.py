from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trivium import synthetic
from trivium.features import efficiency as eff
from trivium.models import profiles
from trivium.storage import read_table


@pytest.fixture(scope="module")
def demo() -> dict[str, pd.DataFrame]:
    return synthetic.demo_tables(scale="small")


# ---------------------------------------------------------------- efficiency index


def test_efficiency_team_match_orientation_and_index(demo: dict[str, pd.DataFrame]) -> None:
    out = eff.efficiency_team_match(demo["team_match"])
    assert len(out) == len(demo["team_match"])
    by_t = out.groupby("tournament")
    assert by_t["cost_measure"].first().to_dict() == {
        "men": "hsr_without_ball_per_dominance", "women": "defensive_work_per_dominance"
    }
    assert np.allclose(by_t["cost_z"].mean(), 0) and np.allclose(by_t["conversion_z"].mean(), 0)
    assert np.allclose(out["efficiency_index"], (out["cost_z"] + out["conversion_z"]) / 2)
    men = out[out["tournament"] == "men"]
    assert men["cost_z"].corr(men["cost_value"], method="spearman") == pytest.approx(-1)


def test_efficiency_falls_back_without_tracking(demo: dict[str, pd.DataFrame]) -> None:
    tm = demo["team_match"].assign(hsr_without_ball_per_dominance=np.nan)
    assert eff.cost_measure(tm, "men") == "defensive_work_per_dominance"
    assert eff.cost_measure(tm.drop(columns="defensive_work_per_dominance"), "men") is None


def test_efficiency_team_tournament_ranks(demo: dict[str, pd.DataFrame]) -> None:
    out = eff.efficiency_team_tournament(demo["team_match"])
    for tournament, t in out.groupby("tournament"):
        n = demo["elo_pre_tournament"].query("tournament == @tournament")["team"].nunique()
        assert len(t) == n and t["efficiency_rank"].min() == 1
        assert t["efficiency_index"].is_monotonic_decreasing


def test_efficiency_position_group_is_aggregated(demo: dict[str, pd.DataFrame]) -> None:
    out = eff.efficiency_position_group(demo["running_player_match"], demo["team_match"])
    assert set(out["position_group"]) == {"Goalkeeper", "Defender", "Midfielder", "Forward"}
    assert "person_id" not in out and out.groupby(["team", "position_group"]).size().max() == 1
    assert eff.efficiency_position_group(None, demo["team_match"]).empty


# ---------------------------------------------------------------- profiles


def test_bootstrap_difference() -> None:
    z = np.r_[np.full(20, 1.0), np.full(20, -1.0)] + np.random.default_rng(0).normal(0, 0.1, 40)
    est, low, high = profiles.bootstrap_difference(z, np.r_[np.ones(20, bool), np.zeros(20, bool)], 500)
    assert est == pytest.approx(2, abs=0.1) and low > 0
    same = profiles.bootstrap_difference(z, np.tile([True, False], 20), 500)
    assert same[1] < 0 < same[2]
    assert profiles.bootstrap_difference(z, np.tile([True, False], 20), 500) == same


def _constructed(n_teams: int = 40) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """dominance tracks Elo (and so stage), conversion tracks over-performance only, field tilt both."""
    rng = np.random.default_rng(1)
    teams = [f"T{i}" for i in range(n_teams)]
    elo = rng.normal(1800, 150, n_teams)
    over = rng.normal(0, 2, n_teams)
    rank = elo.argsort().argsort()
    stage = np.where(rank >= n_teams - n_teams // 3, 3, np.where(rank >= n_teams - 2 * n_teams // 3, 1, 0))
    elo_z = (elo - elo.mean()) / elo.std()
    rows = []
    for i, team in enumerate(teams):
        for match_id in range(3):
            rows.append({"tournament": "men", "match_id": 100 * i + match_id, "team": team,
                         "dominance": 0.5 + (elo[i] - 1800) / 1000 + rng.normal(0, 0.01),
                         "conversion": 2 + 0.5 * np.sign(over[i]) + rng.normal(0, 0.05),
                         "territory_share": 0.5 + 0.05 * (elo_z[i] + 2 * np.sign(over[i])) + rng.normal(0, 0.005)})
    progress = pd.DataFrame({"tournament": "men", "team": teams, "elo": elo, "actual_stage": stage,
                             "actual_stage_name": [["group", "round_of_32", "round_of_16", "quarter_final"][s]
                                                   for s in stage]})
    team_table = pd.DataFrame({"tournament": "men", "team": teams, "overperformance": over})
    return pd.DataFrame(rows), progress, team_table


def test_profile_verdicts() -> None:
    team_match, progress, team_table = _constructed()
    comp = profiles.profile_tables(team_match, progress, team_table, n_boot=500)["profile_comparison"]
    comp = comp.set_index("measure")
    assert comp.loc["dominance", "verdict"] == "explained by strength"
    assert not comp.loc["dominance", "survives_adjustment"] and comp.loc["dominance", "corr_with_elo"] > 0.9
    assert comp.loc["conversion", "verdict"] == "only after adjustment"
    assert comp.loc["territory_share", "verdict"] == "survives adjustment"
    assert comp.loc["territory_share", "survives_adjustment"]
    assert comp.loc["dominance", "n_progressing"] == 40 // 3


@pytest.mark.parametrize(
    "raw, adjusted, same, expected",
    [(True, True, True, "survives adjustment"), (True, True, False, "reverses"),
     (True, False, True, "explained by strength"), (False, True, True, "only after adjustment"),
     (False, False, True, "no clear difference")],
)
def test_verdict(raw: bool, adjusted: bool, same: bool, expected: str) -> None:
    assert profiles.verdict(raw, adjusted, same) == expected


@pytest.fixture(scope="module")
def build(tmp_path_factory: pytest.TempPathFactory):
    cfg = synthetic.write_demo(tmp_path_factory.mktemp("demo"), scale="small", luck_sims=200, progress_sims=200)
    profiles.run(cfg)
    return cfg


def test_profiles_stage_writes_tables_and_figures(build) -> None:
    for name in ("profile_by_stage", "profile_by_overperformance", "profile_comparison", "efficiency_team_match",
                 "efficiency_team_tournament", "efficiency_position_group"):
        assert len(read_table(build, name)) > 0, name
    checks = read_table(build, "checks_profiles")
    failed = checks.loc[~checks["passed"]]
    assert failed["check"].str.contains("sensitivity:").all(), failed["check"].tolist()
    assert (build.build_dir / "figures" / "profile_comparison_men.html").is_file()
    assert (build.build_dir / "profiles.xlsx").is_file()
    by_stage = read_table(build, "profile_by_stage")
    assert set(by_stage.groupby("tournament")["stage_name"].unique().explode()) >= {"group", "champion"}


def test_validation_on_demo_build(build) -> None:
    from trivium import validation

    validation.run(build)
    report = read_table(build, "check_report")
    assert set(report["stage"]) >= {"expectation", "profiles"}
    assert (build.build_dir / "check_report.md").read_text(encoding="utf-8").startswith("# Check report")
    assert "Synthetic demo build" in (build.build_dir / "check_report.md").read_text(encoding="utf-8")


def test_profile_measures_skip_missing_columns() -> None:
    team_match, _, _ = _constructed(6)
    reg = profiles.profile_measures(team_match, "men")
    assert set(reg["measure"]) == {"dominance", "conversion", "territory_share"}
    assert "conversion" not in set(profiles.profile_measures(team_match.assign(conversion=np.nan), "men")["measure"])
