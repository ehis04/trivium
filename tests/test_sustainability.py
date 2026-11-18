from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from trivium import measures as registry
from trivium.models import sustainability as ss
from trivium.storage import read_table, write_table
from trivium.synthetic import demo_config

OUTPUT_COLUMNS = [
    "tournament", "outcome", "term", "coef", "ci_low", "ci_high", "p_value", "n_rows", "n_teams",
    "converged",
]


def _inputs(n_teams: int = 24, n_matches: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Team intercepts plus a known decline and independent nuisance predictors."""
    rng = np.random.default_rng(73)
    team_rows, performance_rows = [], []
    for team_index in range(n_teams):
        team = f"Team {team_index:02d}"
        intercept = rng.normal(0, 0.06)
        for match_number in range(1, n_matches + 1):
            match_id = team_index * n_matches + match_number
            opponent_elo = rng.normal(1700, 150)
            minutes = rng.uniform(0, 500)
            rest_days = np.nan if match_number == 1 else rng.uniform(2, 6)
            dominance = 0.7 + intercept - 0.02 * match_number + rng.normal(0, 0.008)
            team_rows.append({
                "tournament": "men", "match_id": match_id, "team": team,
                "opponent": f"Team {(team_index + 1) % n_teams:02d}",
                "match_number": match_number,
                "starters_minutes_cum": minutes,
                "rest_days": rest_days,
                "dominance": dominance,
                "defensive_work_per_dominance": 3 + 0.1 * match_number + intercept + rng.normal(0, 0.1),
                "hsr_without_ball_km": 4 + 0.05 * match_number + intercept + rng.normal(0, 0.1),
            })
            performance_rows.append({
                "tournament": "men", "match_id": match_id, "team": team,
                "opponent_elo": opponent_elo,
            })
    return pd.DataFrame(team_rows), pd.DataFrame(performance_rows)


def test_rows_join_tournament_keys_and_fill_only_first_match_rest() -> None:
    team_match, performance = _inputs(6, 5)
    later_missing = team_match.index[team_match["match_number"].eq(3)][0]
    team_match.loc[later_missing, "rest_days"] = np.nan
    expected_median = team_match["rest_days"].median()
    women = team_match.assign(tournament="women", dominance=-50)
    women_performance = performance.assign(tournament="women", opponent_elo=9000)
    rows = ss.sustainability_rows(
        pd.concat([team_match, women], ignore_index=True),
        pd.concat([performance, women_performance], ignore_index=True), "men",
    )
    assert len(rows) == len(team_match)
    assert rows["tournament"].eq("men").all()
    assert rows["dominance"].gt(0).all()
    assert rows["opponent_elo"].lt(9000).all()
    first = rows["match_number"].eq(1)
    assert rows.loc[first, "first_match"].eq(1).all()
    assert rows.loc[~first, "first_match"].eq(0).all()
    np.testing.assert_allclose(rows.loc[first, "rest_days"], expected_median)
    missing_key = team_match.loc[later_missing, ["match_id", "team"]]
    later = rows[rows["match_id"].eq(missing_key["match_id"]) & rows["team"].eq(missing_key["team"])]
    assert later["rest_days"].isna().all()


def test_rows_resolve_registry_column_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    team_match, performance = _inputs(6, 5)
    metadata = registry.load_measures()
    aliases = {
        "dominance": "renamed_dominance",
        "match_number": "renamed_match_number",
        "starters_minutes_cum": "renamed_minutes",
        "rest_days": "renamed_rest",
    }
    for name, column in aliases.items():
        metadata.loc[metadata["measure"].eq(name), "column"] = column
    monkeypatch.setattr(registry, "_registry", lambda: metadata.copy())
    monkeypatch.setattr(registry, "load_measures", lambda: metadata.copy())
    rows = ss.sustainability_rows(team_match.rename(columns=aliases), performance, "men")
    assert set(aliases).issubset(rows.columns)
    assert not set(aliases.values()).intersection(rows.columns)
    expected = team_match.sort_values(["tournament", "match_id", "team"], ignore_index=True)
    np.testing.assert_allclose(rows["dominance"], expected["dominance"])
    np.testing.assert_allclose(rows["match_number"], expected["match_number"])


def test_women_rows_exclude_men_only_outcome_even_if_column_is_populated() -> None:
    team_match, performance = _inputs(6, 5)
    rows = ss.sustainability_rows(
        team_match.assign(tournament="women", hsr_without_ball_km=8000),
        performance.assign(tournament="women"), "women",
    )
    assert "hsr_without_ball_km" not in rows or rows["hsr_without_ball_km"].isna().all()
    result = ss.fit_sustainability(rows, "hsr_without_ball_km")
    assert result.empty
    assert result.attrs.get("reason")


def test_mixed_effects_recovers_known_decline_with_interval_below_zero() -> None:
    team_match, performance = _inputs()
    rows = ss.sustainability_rows(team_match, performance, "men")
    fitted = ss.fit_sustainability(rows, "dominance")
    assert list(fitted.columns) == OUTPUT_COLUMNS
    assert not fitted.empty
    match_number = fitted.set_index("term").loc["match_number"]
    assert match_number["coef"] < 0
    assert match_number["ci_high"] < 0
    expected = -0.02 * rows["match_number"].std(ddof=0)
    assert match_number["coef"] == pytest.approx(expected, abs=0.008)
    assert fitted["n_rows"].eq(len(rows)).all()
    assert fitted["n_teams"].eq(24).all()
    assert fitted["converged"].all()
    assert "first_match" in set(fitted["term"])
    assert set(fitted["tournament"]) == {"men"}
    assert set(fitted["outcome"]) == {"dominance"}


@pytest.mark.parametrize("missing", ["dominance", "opponent_elo", "starters_minutes_cum"])
def test_missing_outcome_or_predictor_returns_empty_with_reason(missing: str) -> None:
    team_match, performance = _inputs(6, 5)
    rows = ss.sustainability_rows(team_match, performance, "men").drop(columns=missing)
    result = ss.fit_sustainability(rows, "dominance")
    assert result.empty
    assert list(result.columns) == OUTPUT_COLUMNS
    assert result.attrs.get("reason")


def test_all_missing_outcome_is_skipped() -> None:
    team_match, performance = _inputs(6, 5)
    rows = ss.sustainability_rows(team_match, performance, "men")
    rows["dominance"] = np.nan
    result = ss.fit_sustainability(rows, "dominance")
    assert result.empty
    assert result.attrs.get("reason")


def test_convergence_warnings_are_captured_without_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    team_match, performance = _inputs(8, 5)
    rows = ss.sustainability_rows(team_match, performance, "men")
    original_fit = MixedLM.fit

    def warning_fit(self: MixedLM, *args: object, **kwargs: object) -> object:
        warnings.warn("Synthetic convergence diagnostic.", ConvergenceWarning, stacklevel=2)
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(MixedLM, "fit", warning_fit)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result = ss.fit_sustainability(rows, "dominance")
    assert not result.empty
    assert not recorded
    assert any("Synthetic convergence diagnostic" in message for message in result.attrs["warnings"])
    assert capsys.readouterr().out == ""


def test_singular_fit_failure_is_reported_without_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    team_match, performance = _inputs(6, 5)
    rows = ss.sustainability_rows(team_match, performance, "men")

    def fail_fit(self: MixedLM, *args: object, **kwargs: object) -> object:
        raise np.linalg.LinAlgError("Synthetic singular design.")

    monkeypatch.setattr(MixedLM, "fit", fail_fit)
    result = ss.fit_sustainability(rows, "dominance")
    assert result.empty
    assert result.attrs.get("reason")


def test_stage_writes_results_and_explains_missing_outcomes_and_predictors(tmp_path: Path) -> None:
    men, men_performance = _inputs(8, 5)
    women = men.assign(tournament="women", hsr_without_ball_km=8000)
    women = women.drop(columns="starters_minutes_cum")
    women_performance = men_performance.assign(tournament="women")
    cfg = demo_config(tmp_path)
    write_table(cfg, "team_match", pd.concat([men, women], ignore_index=True))
    write_table(cfg, "overperformance", pd.concat([men_performance, women_performance], ignore_index=True))
    ss.run(cfg)
    result = read_table(cfg, "sustainability")
    checks = read_table(cfg, "checks_sustainability")
    assert list(result.columns) == OUTPUT_COLUMNS
    assert not result.empty
    assert result["tournament"].eq("men").all()
    assert list(checks.columns) == ["check", "target", "value", "passed"]
    assert checks["check"].str.contains("women").any()
    assert checks["check"].str.contains("skip|missing|unavailable", case=False).any()
    assert not (
        result["tournament"].eq("women") & result["outcome"].eq("hsr_without_ball_km")
    ).any()
