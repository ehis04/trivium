from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from three_trees.models import expectation as xp
from three_trees.models import expectation_sensitivity as sens


def _synthetic_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Small two-tournament dataset: men (8 teams) with host and shootout matches,
    and women (8 teams) with no host/shootout matches."""
    rng = np.random.default_rng(20251103)

    # Men's tournament: 8 teams double round robin
    men_teams = [f"M{i}" for i in range(8)]
    men_ratings = dict(zip(men_teams, np.linspace(1400, 2100, 8)))
    men_matches = []
    match_id = 1
    for home, away in permutations(men_teams, 2):
        gap = (men_ratings[home] - men_ratings[away]) / 150
        p_home = 1 / (1 + np.exp(-(gap - 0.6)))
        p_away = 1 / (1 + np.exp(gap + 0.6))
        u = rng.random()
        hs, as_ = (1, 0) if u < p_home else (0, 1) if u < p_home + p_away else (1, 1)
        # Match 1 is host match (home is host)
        neutral = match_id != 1
        # Match 2 and 3 are shootouts (ensure tied score)
        is_so = match_id in (2, 3)
        if is_so:
            hs, as_ = (1, 1)
            so_winner = home
        else:
            so_winner = None
        men_matches.append(
            {
                "tournament": "men",
                "match_id": match_id,
                "match_date": pd.Timestamp("2025-11-03"),
                "competition_stage": "Group Stage",
                "home_team": home,
                "away_team": away,
                "home_score": hs,
                "away_score": as_,
                "neutral_ground": neutral,
                "shootout": is_so,
                "shootout_winner": so_winner,
            }
        )
        match_id += 1

    # Women's tournament: double round robin
    women_teams = [f"W{i}" for i in range(8)]
    women_ratings = dict(zip(women_teams, np.linspace(1400, 2000, 8)))
    women_matches = []
    for home, away in permutations(women_teams, 2):
        gap = (women_ratings[home] - women_ratings[away]) / 150
        p_home = 1 / (1 + np.exp(-(gap - 0.6)))
        p_away = 1 / (1 + np.exp(gap + 0.6))
        u = rng.random()
        hs, as_ = (1, 0) if u < p_home else (0, 1) if u < p_home + p_away else (1, 1)
        women_matches.append(
            {
                "tournament": "women",
                "match_id": match_id,
                "match_date": pd.Timestamp("2025-10-17"),
                "competition_stage": "Group Stage",
                "home_team": home,
                "away_team": away,
                "home_score": hs,
                "away_score": as_,
                "neutral_ground": True,
                "shootout": False,
                "shootout_winner": None,
            }
        )
        match_id += 1

    matches = pd.DataFrame(men_matches + women_matches)
    elo_rows = [{"tournament": "men", "team": t, "elo": men_ratings[t]} for t in men_teams] + [
        {"tournament": "women", "team": t, "elo": women_ratings[t]} for t in women_teams
    ]
    elo_pre = pd.DataFrame(elo_rows)

    tds_team = pd.DataFrame(
        {
            "team": men_teams,
            "best_finish_rank": [1, 2, 3, 4, 5, 6, 7, 8],
            "campaigns_incl_2025": [10, 8, 6, 5, 4, 3, 2, 1],
        }
    )
    return matches, elo_pre, tds_team


def test_main_equals_direct_expectation_computation() -> None:
    matches, elo_pre, tds_team = _synthetic_data()
    v_over = sens.variant_overperformance(matches, elo_pre, tds_team, "main")

    # Direct expectation
    rows = xp.team_match_rows(matches, elo_pre, host_advantage=0.0)
    direct_parts = []
    for _tourney, t_rows in rows.groupby("tournament", sort=False):
        t_rows = t_rows.reset_index(drop=True)
        probs = xp.leave_team_out(t_rows, xp.ordered_logit(["elo_gap"]))
        direct_parts.append(xp.overperformance(t_rows, probs))
    direct = pd.concat(direct_parts, ignore_index=True)

    pd.testing.assert_frame_equal(v_over.drop(columns="variant"), direct)


def test_host_100_equals_main_with_no_host_matches() -> None:
    matches, elo_pre, tds_team = _synthetic_data()
    matches["neutral_ground"] = True

    rows_main, _ = sens.variant_rows(matches, elo_pre, tds_team, "main")
    rows_host, _ = sens.variant_rows(matches, elo_pre, tds_team, "host_100")
    pd.testing.assert_frame_equal(rows_main, rows_host)

    op_main = sens.variant_overperformance(matches, elo_pre, tds_team, "main")
    op_host = sens.variant_overperformance(matches, elo_pre, tds_team, "host_100")
    pd.testing.assert_frame_equal(op_main.drop(columns="variant"), op_host.drop(columns="variant"))


def test_shootout_as_result_changes_shootout_rows_only() -> None:
    matches, elo_pre, tds_team = _synthetic_data()
    rows_main, _ = sens.variant_rows(matches, elo_pre, tds_team, "main")
    rows_so, _ = sens.variant_rows(matches, elo_pre, tds_team, "shootout_as_result")

    # Non-shootout rows are identical
    non_so = ~rows_main["shootout"]
    pd.testing.assert_frame_equal(rows_main[non_so], rows_so[non_so])

    # Shootout rows have win (3.0 pts) for winner, loss (0.0 pts) for loser
    so = rows_main["shootout"]
    assert so.sum() > 0
    winner_mask = so & rows_so["won_shootout"]
    loser_mask = so & (~rows_so["won_shootout"])
    assert (rows_so.loc[winner_mask, "result"] == xp.WIN).all()
    assert (rows_so.loc[winner_mask, "actual_points"] == 3.0).all()
    assert (rows_so.loc[loser_mask, "result"] == xp.LOSS).all()
    assert (rows_so.loc[loser_mask, "actual_points"] == 0.0).all()


def test_no_shootout_matches_removes_shootout_rows_preserves_teams() -> None:
    matches, elo_pre, tds_team = _synthetic_data()
    rows_main, _ = sens.variant_rows(matches, elo_pre, tds_team, "main")
    rows_no_so, _ = sens.variant_rows(matches, elo_pre, tds_team, "no_shootout_matches")

    assert not rows_no_so["shootout"].any()
    assert len(rows_no_so) == len(rows_main[~rows_main["shootout"]])
    assert set(rows_no_so["team"]) == set(rows_main["team"])


def test_tds_variants_return_men_only_and_raise_on_missing_team() -> None:
    matches, elo_pre, tds_team = _synthetic_data()

    for v in ("tds_best_finish", "tds_campaigns", "elo_plus_campaigns"):
        rows, feats = sens.variant_rows(matches, elo_pre, tds_team, v)
        assert (rows["tournament"] == "men").all()
        assert not rows.empty
        if v == "tds_best_finish":
            assert feats == ["tds_best_finish_gap"]
            assert "tds_best_finish_gap" in rows.columns
        elif v == "tds_campaigns":
            assert feats == ["campaigns_gap"]
            assert "campaigns_gap" in rows.columns
        else:
            assert feats == ["elo_gap", "campaigns_gap"]
            assert "campaigns_gap" in rows.columns

    # Missing team raises ValueError
    incomplete_tds = tds_team[tds_team["team"] != "M0"]
    with pytest.raises(ValueError, match="M0"):
        sens.variant_rows(matches, elo_pre, incomplete_tds, "tds_best_finish")


def test_compare_to_main_against_copy_of_itself() -> None:
    matches, elo_pre, tds_team = _synthetic_data()
    team_results, scores = sens.run_variants(matches, elo_pre, tds_team, variants=["main"])
    assert len(scores) == 2  # men and women
    assert set(team_results.columns) == {
        "variant",
        "tournament",
        "team",
        "matches",
        "actual_points",
        "expected_points",
        "overperformance",
        "overperformance_per_match",
    }

    # Clone main as main_copy
    copy_results = team_results.copy()
    copy_results["variant"] = "main_copy"
    combined = pd.concat([team_results, copy_results], ignore_index=True)

    comparison = sens.compare_to_main(combined)
    assert len(comparison) == 2  # men and women
    assert np.allclose(comparison["spearman_total"], 1.0)
    assert np.allclose(comparison["spearman_per_match"], 1.0)
    assert (comparison["sign_changes"] == 0).all()
    assert np.allclose(comparison["max_abs_change"], 0.0)

    # Checks
    checks = sens.checks_sensitivity(comparison)
    assert len(checks) == 2
    assert (checks["passed"]).all()
    assert (checks["target"] == sens.ROBUST_SPEARMAN).all()


def test_elo_formula_fit_properties() -> None:
    # 20% draws in train
    train = pd.DataFrame({"result": [xp.WIN, xp.WIN, xp.DRAW, xp.LOSS, xp.LOSS]})
    test = pd.DataFrame({"elo_gap": [0.0, 400.0, -400.0]})
    probs = sens.elo_formula_fit(train, test)

    # Probabilities sum to 1
    assert np.allclose(probs.sum(axis=1), 1.0)

    # At elo_gap = 0: p_win == p_loss
    assert probs[0, xp.WIN] == pytest.approx(probs[0, xp.LOSS])

    # At elo_gap = +400: p_win + p_draw / 2 == We(400) when not clipped
    # With d = 0.2: We(400) = 1 / 1.1 ≈ 0.90909; p_loss = 1 - 0.90909 - 0.1 = -0.009 (clipped).
    # To test without clipping: use a smaller draw share d = 0.10:
    train_small_draw = pd.DataFrame({"result": [xp.WIN] * 8 + [xp.DRAW] + [xp.LOSS]})
    test_400 = pd.DataFrame({"elo_gap": [400.0]})
    probs_400 = sens.elo_formula_fit(train_small_draw, test_400)
    we_400 = float(sens.expected_score(400.0))
    # p_loss = 1 - 0.90909 - 0.05 = 0.04091 > 0 (unclipped)
    assert probs_400[0, xp.WIN] + probs_400[0, xp.DRAW] / 2.0 == pytest.approx(we_400)


def test_group_stage_fit_equals_main_with_no_knockout_matches() -> None:
    matches, elo_pre, tds_team = _synthetic_data()
    # All matches in _synthetic_data are Group Stage
    op_main = sens.variant_overperformance(matches, elo_pre, tds_team, "main")
    op_group = sens.variant_overperformance(matches, elo_pre, tds_team, "group_stage_fit")
    pd.testing.assert_frame_equal(op_main.drop(columns="variant"), op_group.drop(columns="variant"))


def test_group_stage_fit_with_knockout_matches() -> None:
    matches, elo_pre, tds_team = _synthetic_data()
    # Label some men's matches as Round of 16
    men_matches = matches[matches["tournament"] == "men"].copy()
    men_matches.loc[men_matches["match_id"] > 35, "competition_stage"] = "Round of 16"
    men_elo = elo_pre[elo_pre["tournament"] == "men"]

    team_results, scores = sens.run_variants(
        men_matches, men_elo, tds_team, variants=["main", "group_stage_fit"]
    )
    comp = sens.compare_to_main(team_results, sens.knockout_teams(men_matches))
    assert "spearman_knockout_teams" in comp.columns
    # Knockout spearman should be computed (not nan) for the tournament with knockout matches
    men_comp = comp[(comp["variant"] == "group_stage_fit") & (comp["tournament"] == "men")]
    assert not np.isnan(men_comp["spearman_knockout_teams"].iloc[0])
    assert np.isnan(sens.compare_to_main(team_results)["spearman_knockout_teams"]).all()


def test_knockout_teams_keyed_by_tournament() -> None:
    matches = pd.DataFrame(
        {
            "tournament": ["men", "women", "women"],
            "competition_stage": ["Group Stage", "Round of 16", "Final"],
            "home_team": ["France", "France", "Spain"],
            "away_team": ["Italy", "Japan", "France"],
        }
    )
    assert sens.knockout_teams(matches) == {("women", "France"), ("women", "Japan"), ("women", "Spain")}


def test_scores_summary() -> None:
    # Use synthetic scores table to test pivoting and best_variant logic directly
    scores = pd.DataFrame(
        [
            {"variant": "main", "tournament": "men", "rps": 0.20, "brier": 0.5, "log_loss": 0.9, "accuracy": 0.5},
            {"variant": "host_100", "tournament": "men", "rps": 0.22, "brier": 0.52, "log_loss": 0.92, "accuracy": 0.48},
            {"variant": "elo_formula", "tournament": "men", "rps": 0.18, "brier": 0.48, "log_loss": 0.88, "accuracy": 0.55},
            {"variant": "main", "tournament": "women", "rps": 0.19, "brier": 0.49, "log_loss": 0.89, "accuracy": 0.52},
            {"variant": "host_100", "tournament": "women", "rps": 0.17, "brier": 0.47, "log_loss": 0.87, "accuracy": 0.56},
        ]
    )
    summary = sens.scores_summary(scores)

    assert "tournament" in summary.columns
    assert "best_variant" in summary.columns
    assert "main" in summary.columns
    assert "host_100" in summary.columns
    assert "elo_formula" in summary.columns
    assert len(summary) == 2  # men and women
    men_row = summary[summary["tournament"] == "men"].iloc[0]
    assert men_row["best_variant"] == "elo_formula"
    women_row = summary[summary["tournament"] == "women"].iloc[0]
    assert women_row["best_variant"] == "host_100"

