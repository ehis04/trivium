from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trivium.models import uncertainty as unc


def _table(
    teams: list[str] | None = None,
    n_matches: int = 4,
    probabilities: tuple[float, float, float] = (0.25, 0.25, 0.5),
    actual_points: float = 1.0,
) -> pd.DataFrame:
    teams = teams or ["A"]
    expected_points = probabilities[1] + 3 * probabilities[2]
    return pd.DataFrame(
        [
            {
                "tournament": "men",
                "match_id": match,
                "team": team,
                "p_loss": probabilities[0],
                "p_draw": probabilities[1],
                "p_win": probabilities[2],
                "actual_points": actual_points,
                "expected_points": expected_points,
                "overperformance": actual_points - expected_points,
            }
            for team in teams
            for match in range(n_matches)
        ]
    )


def test_sure_wins_have_zero_bands_and_are_not_beyond_luck() -> None:
    table = _table(n_matches=5, probabilities=(0.0, 0.0, 1.0), actual_points=3.0)
    bands = unc.luck_band(table, n_sims=200, seed=3)

    assert bands.loc[0, ["band_low", "band_high"]].tolist() == [0.0, 0.0]
    assert bands.loc[0, "beyond_luck"] == ""


def test_all_wins_against_low_win_probability_are_beyond_luck() -> None:
    table = _table(n_matches=6, probabilities=(0.8, 0.0, 0.2), actual_points=3.0)
    band = unc.luck_band(table, n_sims=5_000, seed=9).iloc[0]

    assert band["beyond_luck"] == "above"
    assert band["p_upper"] < 0.01
    assert band["overperformance"] == pytest.approx(14.4)


def test_simulated_points_are_valid_and_match_expected_mean() -> None:
    table = _table(n_matches=10, probabilities=(0.45, 0.2, 0.35))
    draws = unc.simulate_points(table, 20_000, np.random.default_rng(42))

    assert set(np.unique(draws)) == {0.0, 1.0, 3.0}
    assert draws.mean(axis=1) == pytest.approx(table["expected_points"], abs=0.05)


def test_luck_band_reproducibility_and_pointwise_bands() -> None:
    table = _table(["A", "B"], n_matches=6)
    first = unc.luck_band(table, n_sims=1_000, seed=4)
    repeated = unc.luck_band(table, n_sims=1_000, seed=4)
    different = unc.luck_band(table, n_sims=1_000, seed=5)

    pd.testing.assert_frame_equal(first, repeated)
    assert not first["band_low"].equals(different["band_low"])
    assert first["matches"].eq(6).all()
    assert first["overperformance_per_match"].mul(first["matches"]).equals(first["overperformance"])


def test_bootstrap_by_team_reports_intervals_and_strata_sizes() -> None:
    rng = np.random.default_rng(22)
    teams = pd.DataFrame(
        {
            "tournament": ["men"] * 30 + ["women"] * 30,
            "team": [f"M{i}" for i in range(30)] + [f"W{i}" for i in range(30)],
            "around_one": rng.normal(1.0, 0.08, 60),
            "around_zero": rng.normal(0.0, 0.08, 60),
        }
    )
    one = unc.bootstrap_by_team(
        teams, lambda sample: sample["around_one"].mean(), n_boot=500, seed=6
    )
    zero = unc.bootstrap_by_team(
        teams, lambda sample: sample["around_zero"].mean(), n_boot=500, seed=6
    )

    assert one["low"] <= 1 <= one["high"]
    assert one["excludes_zero"] == 1.0
    assert zero["excludes_zero"] == 0.0
    assert one["n_boot"] == 500.0

    def fixed_strata_sizes(sample: pd.DataFrame) -> float:
        counts = sample.groupby("tournament").size()
        return float(counts.eq(30).all())

    strata = unc.bootstrap_by_team(teams, fixed_strata_sizes, n_boot=50, seed=8)
    assert strata["low"] == strata["high"] == 1.0


def test_luck_summary_counts_teams_and_expected_chance() -> None:
    bands = pd.DataFrame(
        {
            "tournament": ["men"] * 4 + ["women"] * 2,
            "beyond_luck": ["above", "below", "", "above", "below", ""],
        }
    )
    summary = unc.luck_summary(bands).set_index("tournament")

    assert summary.loc["men", ["teams", "above", "below"]].tolist() == [4, 2, 1]
    assert summary.loc["men", "expected_by_chance"] == pytest.approx(0.2)
    assert summary.loc["women", ["teams", "above", "below"]].tolist() == [2, 0, 1]


def test_uncertainty_checks_have_shared_schema_and_three_rows() -> None:
    table = _table(n_matches=4)
    bands = unc.luck_band(table, n_sims=300, seed=2)
    checks = unc.checks_uncertainty(bands, table, n_sims=300, seed=2)

    assert list(checks.columns) == ["check", "target", "value", "passed"]
    assert len(checks) == 3
    assert checks["check"].str.startswith("men").all()
    assert checks.iloc[2]["passed"]


def test_invalid_probabilities_are_rejected() -> None:
    table = _table()
    table.loc[0, "p_win"] = 0.9

    with pytest.raises(ValueError, match="probabilities"):
        unc.simulate_points(table, 10, np.random.default_rng(1))


def test_calibration_has_equal_count_bins_and_summarises_errors() -> None:
    rng = np.random.default_rng(51)
    n_rows = 10_000
    probs = np.tile([0.4, 0.2, 0.4], (n_rows, 1))
    result = np.array([rng.choice(3, p=row) for row in probs])
    table = pd.DataFrame(
        {
            "tournament": "men",
            "p_loss": probs[:, 0],
            "p_draw": probs[:, 1],
            "p_win": probs[:, 2],
            "result": result,
            "expected_points": probs[:, 1] + 3 * probs[:, 2],
            "actual_points": np.array([0.0, 1.0, 3.0])[result],
        }
    )
    calibration = unc.calibration_table(table, n_bins=5)
    errors = unc.calibration_error(calibration)

    assert calibration["rows"].tolist() == [2_000] * 5
    assert errors.loc[0, "win_probability_error"] <= 0.05
    assert errors.loc[0, "points_error"] <= 0.05

    shifted = table.copy()
    shifted["p_loss"] -= 0.2
    shifted["p_win"] += 0.2
    shifted["expected_points"] += 0.6
    shifted_error = unc.calibration_error(unc.calibration_table(shifted)).loc[0, "points_error"]
    assert shifted_error > errors.loc[0, "points_error"]


def test_calibration_check_is_informational() -> None:
    table = _table(n_matches=3)
    table["result"] = 2
    calibration = unc.calibration_error(unc.calibration_table(table))
    checks = unc.checks_calibration(calibration)

    assert list(checks.columns) == ["check", "target", "value", "passed"]
    assert checks.loc[0, "check"] == "men points calibration error"
    assert np.isnan(checks.loc[0, "target"])
    assert checks.loc[0, "passed"]
