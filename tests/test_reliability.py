from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trivium import measures as registry
from trivium.models import reliability as rel
from trivium.storage import read_table, write_table
from trivium.synthetic import demo_config


def _rows(n_teams: int = 100, n_matches: int = 8, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    effects = np.linspace(-2.0, 2.0, n_teams)
    teams = np.repeat([f"Team {index:03d}" for index in range(n_teams)], n_matches)
    match_days = pd.to_timedelta(np.tile(np.arange(n_matches), n_teams), "D")
    return pd.DataFrame({
        "tournament": "men",
        "team": teams,
        "match_id": np.arange(len(teams)),
        "match_date": pd.Timestamp("2025-11-03") + match_days,
        "signal": np.repeat(effects, n_matches) + rng.normal(0, 0.01, len(teams)),
        "noise": rng.normal(0, 1, len(teams)),
    })


@pytest.mark.parametrize(("r", "expected"), [(0.0, 0.0), (0.5, 2 / 3), (1.0, 1.0), (-0.5, -2.0)])
def test_spearman_brown_known_values(r: float, expected: float) -> None:
    assert rel.spearman_brown(r) == pytest.approx(expected)


@pytest.mark.parametrize("r", [-1.0, np.nan])
def test_spearman_brown_undefined(r: float) -> None:
    assert pd.isna(rel.spearman_brown(r))


@pytest.mark.parametrize("r", [-1.01, 1.01, np.inf])
def test_spearman_brown_rejects_invalid_correlation(r: float) -> None:
    with pytest.raises(ValueError, match="correlation"):
        rel.spearman_brown(r)


def test_team_effect_is_reliable_and_noise_is_not() -> None:
    table = rel.split_half(_rows(500), ["signal", "noise"], "men", random_splits=30).set_index("measure")
    assert table.loc["signal", "r_sb"] > 0.999
    assert table.loc["signal", "r_sb_random_mean"] > 0.999
    assert table.loc["signal", "passed"]
    assert abs(table.loc["noise", "r_sb"]) < 0.2
    assert abs(table.loc["noise", "r_sb_random_mean"]) < 0.2
    assert not table.loc["noise", "passed"]


def test_random_splits_repeat_exactly_after_input_reordering() -> None:
    rows = _rows(30)
    first = rel.split_half(rows, ["signal", "noise"], "men", random_splits=25, seed=11)
    second = rel.split_half(rows.sample(frac=1, random_state=4), ["signal", "noise"], "men",
                            random_splits=25, seed=11)
    pd.testing.assert_frame_equal(first, second)


def test_one_match_and_unobserved_half_do_not_count() -> None:
    rows = _rows(5, 4)
    rows = rows.loc[~rows["team"].eq("Team 000") | rows["match_id"].eq(0)].copy()
    rows.loc[rows["team"].eq("Team 001") & rows["match_id"].isin([5, 7]), "signal"] = np.nan
    table = rel.split_half(rows, ["signal", "noise"], "men", random_splits=10).set_index("measure")
    assert table.loc["signal", "n_teams"] == 3
    assert table.loc["noise", "n_teams"] == 4


def test_constant_and_insufficient_measures_are_undefined_and_fail() -> None:
    rows = _rows(1, 4)
    rows["constant"] = 1.0
    table = rel.split_half(rows, ["signal", "constant"], "men", random_splits=5)
    assert table["r_split"].isna().all()
    assert table["r_sb"].isna().all()
    assert table["r_sb_random_mean"].isna().all()
    assert not table["passed"].any()
    many = _rows(5)
    many["constant"] = 1.0
    assert pd.isna(rel.split_half(many, ["constant"], "men", random_splits=5).iloc[0]["r_sb"])


def test_chronological_dates_then_ids_control_the_split() -> None:
    rows = _rows(4, 4)
    rows["signal"] = np.tile([0.0, 0.0, 0.0, 0.0], 4)
    for team_index in range(4):
        mask = rows["team"].eq(f"Team {team_index:03d}")
        rows.loc[mask, "signal"] = [2 * team_index, -team_index, 3 * team_index, -team_index]
        rows.loc[mask, "match_date"] = pd.to_datetime([
            "2025-11-03", "2025-11-03", "2025-11-02", "2025-11-04",
        ])
    measured = rel.split_half(rows, ["signal"], "men", random_splits=0).iloc[0]
    assert measured["r_split"] == pytest.approx(1.0)
    assert measured["r_sb"] == pytest.approx(1.0)
    assert pd.isna(measured["r_sb_random_mean"])


def test_match_id_fallback_without_dates() -> None:
    rows = _rows(20)
    with_dates = rel.split_half(rows, ["signal"], "men", random_splits=5)
    without_dates = rel.split_half(rows.drop(columns="match_date"), ["signal"], "men", random_splits=5)
    pd.testing.assert_frame_equal(with_dates, without_dates)


def test_reliability_table_joins_dates_and_respects_tournament_keys_and_tiers() -> None:
    men = _rows(8).rename(columns={"signal": "territory_share", "noise": "hsr_km"})
    women = men.assign(tournament="women", territory_share=lambda frame: frame["territory_share"] * -1)
    rows = pd.concat([men, women], ignore_index=True)
    matches = rows[["tournament", "match_id", "match_date"]].drop_duplicates()
    table = rel.reliability_table(rows.drop(columns="match_date"), matches, random_splits=10)
    assert list(table.columns) == rel.RELIABILITY_COLUMNS
    assert set(table["tournament"]) == {"men", "women"}
    assert set(table.loc[table["tournament"].eq("men"), "measure"]) == {"territory_share", "hsr_km"}
    assert table.loc[table["tournament"].eq("women"), "measure"].tolist() == ["territory_share"]
    assert table["n_teams"].eq(8).all()
    assert table.loc[table["measure"].eq("territory_share"), "passed"].all()


def test_registry_renamed_column_reports_semantic_name(monkeypatch: pytest.MonkeyPatch) -> None:
    registered = registry.load_measures()
    registered.loc[registered["measure"].eq("territory_share"), "column"] = "tilt_from_provider"
    monkeypatch.setattr(registry, "load_measures", lambda: registered.copy())
    rows = _rows(10).rename(columns={"signal": "tilt_from_provider"})
    matches = rows[["tournament", "match_id", "match_date"]]
    table = rel.reliability_table(rows.drop(columns="match_date"), matches, random_splits=5)
    assert table["measure"].tolist() == ["territory_share"]
    assert table["label"].tolist() == ["Field tilt"]
    assert rel.reliable_measures(table, "men") == ["territory_share"]
    physical = rel.split_half(rows, ["tilt_from_provider"], "men", random_splits=5)
    pd.testing.assert_frame_equal(table, physical)


def test_reliable_measures_keeps_only_allowed_passed_signifiers() -> None:
    table = pd.DataFrame({
        "tournament": ["men", "men", "men", "women", "women", "women"],
        "measure": [
            "territory_share", "possession_share", "security", "territory_share", "hsr_km", "security",
        ],
        "passed": [True, True, False, True, True, False],
    })
    assert rel.reliable_measures(table, "men") == ["territory_share"]
    assert rel.reliable_measures(table, "men", signifiers_only=False) == [
        "territory_share", "possession_share",
    ]
    assert rel.reliable_measures(table, "women", signifiers_only=False) == ["territory_share"]


def test_direct_split_skips_men_only_measures_for_women() -> None:
    rows = _rows(5).rename(columns={"signal": "hsr_km"}).assign(tournament="women")
    assert rel.split_half(rows, ["hsr_km"], "women", random_splits=5).empty


def test_empty_tournament_retains_return_schema() -> None:
    table = rel.split_half(_rows(5), ["signal"], "women", random_splits=5)
    assert list(table.columns) == rel.RELIABILITY_COLUMNS
    assert table.iloc[0]["n_teams"] == 0
    assert not table.iloc[0]["passed"]


def test_bad_dates_and_duplicate_team_match_rows_raise() -> None:
    rows = _rows(5)
    bad = rows.copy()
    bad.loc[0, "match_date"] = pd.NaT
    with pytest.raises(ValueError, match="dates"):
        rel.split_half(bad, ["signal"], "men")
    with pytest.raises(ValueError, match="one row per team"):
        rel.split_half(pd.concat([rows, rows.iloc[:1]], ignore_index=True), ["signal"], "men")
    matches = rows[["tournament", "match_id", "match_date"]]
    with pytest.raises(ValueError, match="dates"):
        rel.reliability_table(rows, matches.iloc[1:], random_splits=5)


@pytest.mark.parametrize("random_splits", [-1, 1.5, True])
def test_invalid_random_split_count_raises(random_splits: int) -> None:
    with pytest.raises(ValueError, match="random splits"):
        rel.split_half(_rows(5), ["signal"], "men", random_splits=random_splits)


def test_run_writes_reliability_and_all_signifier_checks(tmp_path: Path) -> None:
    cfg = demo_config(tmp_path)
    rows = _rows(8).rename(columns={"signal": "territory_share"})
    write_table(cfg, "team_match", rows.drop(columns="match_date"))
    write_table(cfg, "matches", rows[["tournament", "match_id", "match_date"]])
    rel.run(cfg)
    table = read_table(cfg, "reliability")
    checks = read_table(cfg, "checks_reliability")
    assert table["measure"].tolist() == ["territory_share"]
    assert table.iloc[0]["passed"]
    assert len(checks) == len(registry.measures(tournament="men", table="team_match", signifiers_only=True))
    assert list(checks.columns) == rel.CHECK_COLUMNS
    assert checks["check"].str.startswith("men").all()
    assert checks["target"].eq(0.5).all()
    assert checks["passed"].sum() == 1
