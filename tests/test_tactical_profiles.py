from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from three_trees.models import tactical_profiles as tp


def _team_tournament(rng: np.random.Generator, n: int = 30) -> pd.DataFrame:
    return pd.DataFrame({
        "tournament": "men", "team": [f"T{i}" for i in range(n)], "matches": 3,
        "dominance": rng.normal(0.5, 0.05, n), "conversion": rng.normal(1.0, 0.3, n),
        "security": rng.normal(0.05, 0.01, n), "defensive_work_per_dominance": rng.normal(3.0, 0.5, n),
        "hsr_without_ball_per_dominance": rng.normal(80, 10, n),
    })


def test_scores_are_oriented_and_profiles_named() -> None:
    tt = _team_tournament(np.random.default_rng(1))
    tt.loc[0, ["dominance", "conversion", "security", "defensive_work_per_dominance",
               "hsr_without_ball_per_dominance"]] = [0.7, 2.0, 0.01, 1.5, 50]
    scores = tp.axis_scores(tt).set_index("team")
    assert (scores.loc["T0", ["dominant", "efficient", "resourceful"]] > 1).all()
    assert scores[["dominant", "efficient", "resourceful"]].std().round(6).eq(1).all()
    assert tp.profile_name(scores.loc["T0"]) == "Dominant · Efficient · Resourceful"
    women = tt.assign(tournament="women")
    assert tp.axis_scores(women)["resourceful"].notna().all()  # women: defensive work only


def test_summary_and_axis_effects() -> None:
    rng = np.random.default_rng(2)
    tt = _team_tournament(rng, 40)
    scores = tp.axis_scores(tt)
    team_table = scores[["tournament", "team"]].assign(
        overperformance_per_match=0.5 * scores["efficient"] + rng.normal(0, 0.2, 40))
    tables = tp.tactical_profile_tables(tt, team_table, n_boot=300)
    summary, axes = tables["tactical_profile_summary"], tables["tactical_profile_axes"].set_index("axis")
    assert summary["n_teams"].sum() == 40 and summary["overperformance_per_match"].is_monotonic_decreasing
    assert axes.loc["efficient", "excludes_zero"] and axes.loc["efficient", "effect"] == pytest.approx(0.5, abs=0.15)
    assert axes.loc["efficient", "spearman_low"] > 0 and axes.loc["efficient", "spearman_excludes_zero"]


def test_match_scores_reliability_and_early_late() -> None:
    rng = np.random.default_rng(3)
    teams = [f"T{i}" for i in range(20)]
    level = dict(zip(teams, rng.normal(0, 1, 20)))
    rows = []
    for m in range(6):
        for t in teams:
            rows.append({"tournament": "men", "match_id": m * 100 + teams.index(t), "team": t,
                         "dominance": 0.5 + 0.05 * level[t] + rng.normal(0, 0.01), "conversion": 1 + level[t],
                         "security": 0.05, "defensive_work_per_dominance": 3 - level[t]})
    tm = pd.DataFrame(rows)
    scores = tp.match_scores(tm)
    rel = tp.axis_reliability(scores).set_index("axis")
    assert rel.loc["dominant", "repeatable"]
    op = tm[["tournament", "match_id", "team"]].assign(
        match_date=pd.to_datetime("2025-11-01") + pd.to_timedelta(tm["match_id"] // 100, unit="D"),
        overperformance=tm["team"].map(level) + rng.normal(0, 0.1, len(tm)))
    later = tp.early_late(scores, op, n_boot=300).set_index("axis")
    assert later.loc["dominant", "predicts_later"] and later.loc["dominant", "early_late_spearman"] > 0.8
