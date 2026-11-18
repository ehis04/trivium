from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import RidgeCV

from three_trees.models import drivers as dr
from three_trees.storage import read_table, write_table
from three_trees.synthetic import demo_config


def _rows(n_teams: int = 20, seed: int = 32) -> pd.DataFrame:
    """Mirrored team-match rows with one known useful feature and one noise feature."""
    rng = np.random.default_rng(seed)
    teams = [f"Team {index:02d}" for index in range(n_teams)]
    records = []
    for match_id, (home, away) in enumerate(combinations(teams, 2), start=1):
        a, b, error = rng.normal(size=3)
        for team, opponent, sign in ((home, away, 1), (away, home, -1)):
            records.append({
                "tournament": "men", "match_id": match_id, "team": team, "opponent": opponent,
                "feature_a": sign * a, "feature_b": sign * b,
                "overperformance": sign * (0.5 * a + 0.15 * error),
                "team_elo": 1500 + 20 * teams.index(team),
            })
    rows = pd.DataFrame(records)
    x = rows[["feature_a", "feature_b"]].to_numpy()
    error = rows["overperformance"].to_numpy() - 0.5 * x[:, 0]
    error -= x @ np.linalg.lstsq(x, error, rcond=None)[0]
    rows["overperformance"] = 0.5 * x[:, 0] + error
    return rows


def _split_inputs(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = rows[[*dr.KEYS, "feature_a", "feature_b"]].copy()
    target = rows[[*dr.KEYS, "overperformance", "team_elo"]].copy()
    return features, target


def test_driver_rows_joins_on_tournament_match_and_team_and_counts_drops() -> None:
    rows = _rows(8)
    features, target = _split_inputs(rows)
    women_features = features.assign(tournament="women", feature_a=100.0)
    women_target = target.assign(tournament="women", overperformance=200.0)
    features.loc[0, "feature_a"] = np.nan
    target.loc[1, "overperformance"] = np.nan
    result = dr.driver_rows(
        pd.concat([features, women_features], ignore_index=True),
        pd.concat([target, women_target], ignore_index=True),
        ["feature_a", "feature_b"], "men",
    )
    assert len(result) == len(rows) - 2
    assert result["tournament"].eq("men").all()
    assert result.attrs["n_rows_input"] == len(rows)
    assert result.attrs["n_rows_dropped"] == 2
    assert result["feature_a"].abs().max() < 100


def test_driver_rows_resolves_registry_source_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _rows(8)
    features, target = _split_inputs(rows)
    metadata = dr.registry.load_measures()
    metadata.loc[metadata["measure"].eq("territory_share"), "column"] = "renamed_field_tilt"
    monkeypatch.setattr(dr.registry, "load_measures", lambda: metadata.copy())
    features = features.rename(columns={"feature_a": "renamed_field_tilt"})
    result = dr.driver_rows(features, target, ["territory_share"], "men")
    assert "territory_share" in result
    assert "renamed_field_tilt" not in result
    np.testing.assert_allclose(result["territory_share"], rows.sort_values(dr.KEYS)["feature_a"])


def test_driver_rows_rejects_duplicate_match_sides() -> None:
    features, target = _split_inputs(_rows(8))
    with pytest.raises(ValueError, match="one row"):
        dr.driver_rows(pd.concat([features, features.iloc[[0]]]), target, ["feature_a"], "men")


def test_cross_validation_holds_out_teams_and_both_sides_of_matches() -> None:
    cv = dr.cross_validate(_rows(), ["feature_a", "feature_b"], n_folds=5)
    assert len(cv) == 5
    assert cv["n_test_rows"].sum() == len(_rows())
    for fold in cv.attrs["folds"]:
        assert not set(fold["train_match_ids"]) & set(fold["test_match_ids"])
        assert not set(fold["train_teams"]) & set(fold["test_teams"])
    assert cv.attrs["leakage_violations"] == 0
    assert cv["mse_model"].mean() < cv["mse_baseline"].mean()


def test_cross_validation_scales_only_training_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _rows()
    rows.loc[rows["team"].eq("Team 00"), "feature_a"] += 10_000
    seen = []

    class RecordingRidge(RidgeCV):
        def fit(self, x: np.ndarray, y: np.ndarray) -> RecordingRidge:
            seen.append(x.copy())
            return super().fit(x, y)

    monkeypatch.setattr(dr, "RidgeCV", RecordingRidge)
    dr.cross_validate(rows, ["feature_a", "feature_b"])
    assert len(seen) == 5
    for training in seen:
        np.testing.assert_allclose(training.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(training.std(axis=0), 1, atol=1e-12)


def test_fit_drivers_recovers_informative_feature_and_noise_interval() -> None:
    rows = _rows()
    coefficients, summary = dr.fit_drivers(rows, ["feature_a", "feature_b"], n_boot=300, seed=42)
    fitted = coefficients.set_index("measure")
    assert fitted.loc["feature_a", "excludes_zero"]
    assert fitted.loc["feature_a", "coef_low"] > 0
    assert not fitted.loc["feature_b", "excludes_zero"]
    assert summary["beats_baseline"]
    assert summary["oos_r2"] > 0.8
    assert fitted.loc["feature_a", "importance"] > fitted.loc["feature_b", "importance"]
    assert fitted.loc["feature_a", "univariate_low"] > 0
    assert fitted.loc["feature_a", "coef"] == pytest.approx(0.5 * rows["feature_a"].std(ddof=0), abs=0.03)
    assert fitted.loc["feature_a", "feature_mean"] == pytest.approx(rows["feature_a"].mean())
    assert fitted.loc["feature_a", "feature_sd"] == pytest.approx(rows["feature_a"].std(ddof=0))


def test_fit_drivers_same_seed_repeats_exactly() -> None:
    rows = _rows(12)
    first, first_summary = dr.fit_drivers(rows, ["feature_a", "feature_b"], n_boot=50, seed=11)
    second, second_summary = dr.fit_drivers(rows, ["feature_a", "feature_b"], n_boot=50, seed=11)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_series_equal(pd.Series(first_summary), pd.Series(second_summary))


def test_no_features_returns_explicit_null_summary_and_zero_baseline() -> None:
    rows = _rows(12)
    drivers, summary = dr.fit_drivers(rows, [], n_boot=10)
    cv = dr.cross_validate(rows, [])
    assert drivers.empty
    assert list(drivers.columns) == dr.DRIVER_COLUMNS
    assert summary["signifiers_used"] == 0
    assert not summary["beats_baseline"]
    assert summary["oos_r2"] == 0
    assert cv["mse_model"].equals(cv["mse_baseline"])
    assert cv["alpha"].isna().all()


def test_empty_or_tiny_purged_training_sets_use_baseline() -> None:
    rows = _rows(2)
    cv = dr.cross_validate(rows, ["feature_a"])
    assert len(cv) == 2
    assert cv["mse_model"].equals(cv["mse_baseline"])
    assert cv["alpha"].isna().all()
    empty, summary = dr.fit_drivers(rows.iloc[:0], [], n_boot=10)
    assert empty.empty
    assert summary["n_rows"] == 0
    assert np.isnan(summary["oos_r2"])


def test_one_team_cannot_produce_driver_intervals() -> None:
    rows = _rows(8).loc[lambda table: table["team"].eq("Team 00")].copy()
    rows["overperformance"] = 0.5 * rows["feature_a"]
    drivers, summary = dr.fit_drivers(rows, ["feature_a"], n_boot=20)
    assert drivers.empty
    assert list(drivers.columns) == dr.DRIVER_COLUMNS
    assert summary["n_rows"] == len(rows)
    assert summary["n_teams"] == 1
    assert summary["signifiers_used"] == 1
    assert np.isnan(summary["alpha"])
    assert np.isnan(summary["oos_r2"])
    assert not summary["beats_baseline"]


def test_fit_rejects_mixed_tournaments() -> None:
    rows = _rows(8)
    rows.loc[0, "tournament"] = "women"
    with pytest.raises(ValueError, match="one tournament"):
        dr.fit_drivers(rows, ["feature_a"], n_boot=10)


@pytest.mark.parametrize("n_boot", [0, -1, 1.5, True])
def test_fit_rejects_bad_bootstrap_counts(n_boot: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        dr.fit_drivers(_rows(8), ["feature_a"], n_boot=n_boot)


def test_stage_writes_null_result_and_women_tier_filter(tmp_path: Path) -> None:
    men = _rows(10)
    women = men.assign(tournament="women")
    rows = pd.concat([men, women], ignore_index=True)
    features, target = _split_inputs(rows)
    features["hsr_without_ball_per_dominance"] = rows["feature_a"]
    reliability = pd.DataFrame({
        "tournament": ["men", "women"],
        "measure": ["hsr_without_ball_per_dominance"] * 2,
        "passed": [True, True],
    })
    cfg = demo_config(tmp_path)
    write_table(cfg, "team_match", features)
    write_table(cfg, "overperformance", target)
    write_table(cfg, "reliability", reliability)
    dr.run(cfg)
    summary = read_table(cfg, "drivers_summary").set_index("tournament")
    assert summary.loc["men", "signifiers_used"] == 1
    assert summary.loc["women", "signifiers_used"] == 0
    assert not summary.loc["women", "beats_baseline"]
    coefficients = read_table(cfg, "drivers")
    assert coefficients["tournament"].eq("men").all()
    checks = read_table(cfg, "checks_drivers")
    leakage = checks["check"].str.contains("held-out matches present")
    assert checks.loc[leakage, "passed"].all()
    assert checks.loc[leakage, "value"].eq(0).all()
    assert set(read_table(cfg, "drivers_cv")["tournament"]) == {"men", "women"}
