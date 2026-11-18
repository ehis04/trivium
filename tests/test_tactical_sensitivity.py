from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from three_trees.charts.tactical import sensitivity_chart
from three_trees.models import tactical_profiles as tp


def _sensitivity_frames(n_teams: int = 40) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(814)
    teams = [f"T{i}" for i in range(n_teams)]
    signal = rng.normal(size=n_teams)
    elo = np.linspace(1400, 2100, n_teams)
    matches, outcomes = [], []
    for i, team in enumerate(teams):
        for match in range(3):
            group = match < 2
            matches.append({
                "tournament": "men", "match_id": i * 10 + match, "team": team,
                "dominance": signal[i] + rng.normal(0, 0.04),
                "contests_share": signal[i] + rng.normal(0, 0.04),
                "territory_share": rng.normal(0, 1),
                "chances_share": rng.normal(0, 1),
                "conversion": rng.normal(0, 1), "security": rng.normal(0, 1),
                "defensive_work_per_dominance": rng.normal(0, 1),
                "hsr_without_ball_per_dominance": rng.normal(0, 1),
            })
            outcomes.append({
                "tournament": "men", "match_id": i * 10 + match, "team": team,
                "competition_stage": "Group Stage" if group else "Quarter Final",
                "overperformance": signal[i] + rng.normal(0, 0.2),
            })
    team_match = pd.DataFrame(matches)
    overperformance = pd.DataFrame(outcomes)
    team_tournament = team_match.groupby(["tournament", "team"], as_index=False).mean(numeric_only=True)
    team_tournament["matches"] = 3
    team_table = overperformance.groupby(["tournament", "team"], as_index=False).agg(
        overperformance_per_match=("overperformance", "mean")
    )
    team_table["elo"] = team_table["team"].map(dict(zip(teams, elo)))
    return team_tournament, team_table, team_match, overperformance


def test_sensitivity_variants_compare_parts_and_elo_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tp, "N_BOOT", 300)
    team_tournament, team_table, team_match, overperformance = _sensitivity_frames()
    sensitivity = tp.profile_sensitivity(team_tournament, team_table, team_match, overperformance)
    rows = sensitivity.set_index(["score", "variant"])

    contests = rows.loc[("dominant", "contests_share")]
    assert contests["spearman"] > 0.8 and contests["low"] > 0
    assert contests["same_sign_as_main"]

    group = rows.loc[("dominant", "group_stage_only")]
    assert group["spearman"] > 0.8
    assert group["n_teams"] == 40

    without_strongest = rows.loc[("dominant", "without_strongest")]
    assert without_strongest["n_teams"] == 30
    main_scores = tp.axis_scores(team_tournament).set_index("team")["dominant"]
    retained = team_table.nsmallest(30, "elo")
    expected = retained["team"].map(main_scores).corr(retained["overperformance_per_match"], method="spearman")
    assert without_strongest["spearman"] == pytest.approx(expected)


def test_group_stage_variant_uses_group_matches_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tp, "N_BOOT", 300)
    team_tournament, team_table, team_match, overperformance = _sensitivity_frames()
    signal = team_table.set_index("team")["overperformance_per_match"]
    group = overperformance["competition_stage"].str.contains("group", case=False)
    team_match["dominance"] = team_match["team"].map(signal) * np.where(group, 1, -3)
    team_tournament = team_match.groupby(["tournament", "team"], as_index=False).mean(numeric_only=True)
    team_tournament["matches"] = 3

    sensitivity = tp.profile_sensitivity(team_tournament, team_table, team_match, overperformance)
    dominant = sensitivity[sensitivity["score"] == "dominant"].set_index("variant")
    assert dominant.loc["group_stage_only", "spearman"] > 0.8
    assert dominant.loc["main", "spearman"] < -0.8


def test_women_have_no_tracking_only_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tp, "N_BOOT", 100)
    team_tournament, team_table, team_match, overperformance = _sensitivity_frames()
    sensitivity = tp.profile_sensitivity(
        team_tournament.assign(tournament="women"),
        team_table.assign(tournament="women"),
        team_match.assign(tournament="women"),
        overperformance.assign(tournament="women"),
    )
    assert "hsr_without_ball_per_dominance" not in set(sensitivity["variant"])


def test_profile_stability_is_one_when_profiles_do_not_change() -> None:
    values = np.r_[np.arange(-8, 0), np.arange(1, 9)].astype(float)
    teams = pd.DataFrame({
        "tournament": "men",
        "team": [f"T{i}" for i in range(len(values))],
        "matches": 3,
        "dominance": 0.5 + values / 100,
        "conversion": 1 + values,
        "security": 0.05 - values / 100,
        "defensive_work_per_dominance": 3 - values,
        "hsr_without_ball_per_dominance": 80 - values,
    })
    stability = tp.profile_stability(teams).iloc[0]
    assert stability["median_threshold_share"] == 1.0
    assert stability["leave_one_out_share"] == 1.0


def test_sensitivity_chart_has_intervals_zero_line_and_highlighted_main() -> None:
    table = pd.DataFrame({
        "tournament": ["men"] * 3,
        "score": ["dominant"] * 3,
        "variant": ["main", "contests_share", "group_stage_only"],
        "n_teams": [40] * 3,
        "spearman": [0.4, 0.5, 0.2],
        "low": [0.1, 0.2, -0.1],
        "high": [0.7, 0.8, 0.5],
        "same_sign_as_main": [True] * 3,
    })
    fig = sensitivity_chart(table, "men")
    assert len(fig.data) == 1
    assert len(fig.data[0].x) == 3
    assert list(fig.data[0].error_x.array) == pytest.approx([0.3, 0.3, 0.3])
    assert fig.data[0].marker.symbol[0] == "diamond"
    assert any(shape.type == "line" and shape.x0 == 0 for shape in fig.layout.shapes)


def test_profiles_stage_writes_sensitivity_table(tmp_path) -> None:
    from three_trees import synthetic
    from three_trees.models import profiles
    from three_trees.storage import read_table

    cfg = synthetic.write_demo(tmp_path / "demo", scale="small", luck_sims=100, progress_sims=100)
    profiles.run(cfg)
    table = read_table(cfg, "tactical_profile_sensitivity")
    assert {"score", "variant", "spearman", "low", "high", "same_sign_as_main",
            "median_threshold_share", "leave_one_out_share"} <= set(table.columns)
    checks = read_table(cfg, "checks_profiles")
    assert checks["check"].str.contains("sensitivity: dominant keeps its sign in every variant").any()
    assert checks["check"].str.contains("sensitivity: profile assignment stability").any()
