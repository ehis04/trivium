from __future__ import annotations

from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trivium.config import DATA_ROOT_ENV, load_config
from trivium.models import expectation as xp
from trivium.storage import read_table, write_table

EXAMPLE = Path(__file__).resolve().parents[1] / "config.example.yaml"


def _tournament(n_teams: int = 10, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Double round robin with results drawn from the Elo gap."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    ratings = dict(zip(teams, np.linspace(1400, 2100, n_teams)))
    rows = []
    for match_id, (home, away) in enumerate(permutations(teams, 2), start=1):
        gap = (ratings[home] - ratings[away]) / 150
        p_home = 1 / (1 + np.exp(-(gap - 0.6)))
        p_away = 1 / (1 + np.exp(gap + 0.6))
        u = rng.random()
        hs, as_ = (1, 0) if u < p_home else (0, 1) if u < p_home + p_away else (1, 1)
        rows.append((match_id, home, away, hs, as_))
    matches = pd.DataFrame(rows, columns=["match_id", "home_team", "away_team", "home_score", "away_score"])
    matches = matches.assign(
        tournament="men", match_date=pd.Timestamp("2025-11-03"), competition_stage="Group Stage",
        neutral_ground=True, shootout=False, shootout_winner=None,
    )
    elo = pd.DataFrame({"tournament": "men", "team": teams, "elo": [ratings[t] for t in teams]})
    return matches, elo


def test_team_match_rows_two_views_host_and_shootout() -> None:
    matches = pd.DataFrame(
        {
            "tournament": "men",
            "match_id": [1, 2],
            "match_date": pd.Timestamp("2025-11-03"),
            "competition_stage": ["Group Stage", "Round of 32"],
            "home_team": ["A", "C"],
            "away_team": ["B", "D"],
            "home_score": [2, 1],
            "away_score": [1, 1],
            "neutral_ground": [False, True],
            "shootout": [False, True],
            "shootout_winner": [None, "D"],
        }
    )
    elo = pd.DataFrame({"tournament": "men", "team": ["A", "B", "C", "D"], "elo": [1600.0, 1500, 1500, 1500]})
    rows = xp.team_match_rows(matches, elo, host_advantage=100).set_index("team")

    assert len(rows) == 4
    assert rows.loc["A", "elo_gap"] == 200
    assert rows.loc["B", "elo_gap"] == -200
    assert rows.loc["A", "result"] == xp.WIN and rows.loc["B", "result"] == xp.LOSS
    assert rows.loc["D", "result"] == xp.DRAW and rows.loc["D", "actual_points"] == 1
    assert rows.loc["D", "won_shootout"] and not rows.loc["C", "won_shootout"]
    assert xp.team_match_rows(matches, elo).set_index("team").loc["A", "elo_gap"] == 100


def test_leave_team_out_never_trains_on_own_matches() -> None:
    rows = xp.team_match_rows(*_tournament())
    seen = {}

    def spy(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        team = test["team"].unique()
        assert len(team) == 1
        seen[team[0]] = set(train["match_id"])
        return xp.base_rates(train, test)

    probs = xp.leave_team_out(rows, spy)
    assert not probs.isna().any().any()
    for team, train_ids in seen.items():
        assert not train_ids & set(rows.loc[rows["team"] == team, "match_id"])
    assert set(seen) == set(rows["team"])


def test_leave_team_out_one_tournament_only() -> None:
    rows = xp.team_match_rows(*_tournament())
    rows.loc[0, "tournament"] = "women"
    with pytest.raises(ValueError):
        xp.leave_team_out(rows, xp.base_rates)


def test_ordered_logit_monotonic_and_symmetric() -> None:
    rows = xp.team_match_rows(*_tournament())
    test = pd.DataFrame({"elo_gap": [-400.0, 0.0, 400.0]})
    probs = xp.ordered_logit()(rows, test)

    assert probs.sum(axis=1) == pytest.approx(np.ones(3))
    assert probs[0, xp.WIN] < probs[1, xp.WIN] < probs[2, xp.WIN]
    assert probs[2, xp.WIN] == pytest.approx(probs[0, xp.LOSS], abs=1e-3)
    assert probs[1, xp.WIN] == pytest.approx(probs[1, xp.LOSS], abs=1e-3)


def test_ordered_logit_beats_base_rates_on_informative_data() -> None:
    rows = xp.team_match_rows(*_tournament())
    scores = xp.compare_models(rows, {"base": xp.base_rates, "logit": xp.ordered_logit()}).set_index("model")
    assert scores.loc["logit", "rps"] < scores.loc["base", "rps"]
    assert scores.loc["logit", "log_loss"] < scores.loc["base", "log_loss"]


def test_score_predictions_reference_values() -> None:
    result = np.array([xp.LOSS, xp.DRAW, xp.WIN])
    perfect = xp.score_predictions(result, np.eye(3)[result])
    assert perfect["brier"] == 0 and perfect["rps"] == 0 and perfect["accuracy"] == 1
    assert perfect["log_loss"] == pytest.approx(0, abs=1e-9)

    uniform = xp.score_predictions(np.array([xp.DRAW]), np.full((1, 3), 1 / 3))
    assert uniform["brier"] == pytest.approx(6 / 9)
    assert uniform["rps"] == pytest.approx(1 / 9)
    assert uniform["log_loss"] == pytest.approx(np.log(3))


def test_overperformance_and_team_totals() -> None:
    rows = xp.team_match_rows(*_tournament(n_teams=4))
    probs = pd.DataFrame(np.tile([0.25, 0.25, 0.5], (len(rows), 1)), columns=xp.PROB_COLUMNS)
    table = xp.overperformance(rows, probs)

    assert table["expected_points"].eq(1.75).all()
    assert (table["overperformance"] == table["actual_points"] - 1.75).all()

    teams = xp.team_overperformance(table)
    assert len(teams) == 4
    assert teams["matches"].eq(6).all()
    assert teams["overperformance"].sum() == pytest.approx(table["overperformance"].sum())
    assert teams["overperformance_per_match"].mul(teams["matches"]).sum() == pytest.approx(
        table["overperformance"].sum()
    )


def _two_tournaments() -> tuple[pd.DataFrame, pd.DataFrame]:
    men, men_elo = _tournament(n_teams=10, seed=7)
    men.loc[men.index[-10:], "competition_stage"] = "Round of 16"
    women, women_elo = _tournament(n_teams=8, seed=11)
    women = women.assign(tournament="women", match_id=women["match_id"] + 1000)
    rename = {f"T{i}": f"W{i}" for i in range(8)}
    women[["home_team", "away_team"]] = women[["home_team", "away_team"]].replace(rename)
    women_elo = women_elo.assign(tournament="women", team=women_elo["team"].replace(rename))
    return pd.concat([men, women], ignore_index=True), pd.concat([men_elo, women_elo], ignore_index=True)


def test_expectation_tables_shapes_and_checks() -> None:
    matches, elo = _two_tournaments()
    tables = xp.expectation_tables(matches, elo)

    assert len(tables["overperformance"]) == 2 * len(matches)
    assert list(tables["expected_results"].columns) == xp.EXPECTED_RESULTS_COLUMNS
    assert set(tables["overperformance_team"]["tournament"]) == {"men", "women"}

    checks = xp.checks_expectation(tables["overperformance"])
    assert checks["passed"].all(), checks.loc[~checks["passed"], "check"].tolist()
    assert checks["check"].str.contains("draw share, knockout").sum() == 2


def test_expectation_tables_refuse_missing_elo() -> None:
    matches, elo = _two_tournaments()
    with pytest.raises(ValueError, match="T3"):
        xp.expectation_tables(matches, elo[elo["team"] != "T3"])


KNOCKOUT_LABELS = {16: "Round of 16", 8: "Quarter-finals", 4: "Semi-finals", 2: "Final"}


def _cup(tournament: str, n_groups: int, first_round: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Groups of four, best thirds, single-elimination knockout with shootouts and a third-place match."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(4 * n_groups)]
    ratings = dict(zip(teams, rng.normal(1800, 150, len(teams))))
    rows: list[tuple] = []

    def play(home: str, away: str, stage: str, knockout: bool) -> str | None:
        gap = (ratings[home] - ratings[away]) / 150
        p_home, p_away = 1 / (1 + np.exp(-(gap - 0.6))), 1 / (1 + np.exp(gap + 0.6))
        u = rng.random()
        hs, as_ = (1, 0) if u < p_home else (0, 1) if u < p_home + p_away else (1, 1)
        winner = (home if rng.random() < 0.5 else away) if knockout and hs == as_ else None
        rows.append((len(rows) + 1, stage, home, away, hs, as_, winner is not None, winner))
        return winner or (home if hs > as_ else away)

    points = dict.fromkeys(teams, 0.0)
    thirds = []
    qualifiers = []
    for g in range(n_groups):
        group = teams[4 * g: 4 * g + 4]
        for i, j in [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)]:
            play(group[i], group[j], "Group Stage", False)
            hs, as_ = rows[-1][4], rows[-1][5]
            points[group[i]] += 3 if hs > as_ else 1 if hs == as_ else 0
            points[group[j]] += 3 if as_ > hs else 1 if hs == as_ else 0
        ranked = sorted(group, key=lambda t: (points[t], ratings[t]), reverse=True)
        qualifiers += ranked[:2]
        thirds.append(ranked[2])
    qualifiers += sorted(thirds, key=lambda t: (points[t], ratings[t]), reverse=True)[: first_round - 2 * n_groups]

    alive = qualifiers
    while len(alive) > 1:
        label = KNOCKOUT_LABELS[len(alive)]
        pairs = [(alive[i], alive[-1 - i]) for i in range(len(alive) // 2)]
        winners = [play(a, b, label, True) for a, b in pairs]
        if len(alive) == 4:
            losers = [b if w == a else a for (a, b), w in zip(pairs, winners)]
            play(losers[0], losers[1], "3rd Place Final", True)
        alive = winners

    matches = pd.DataFrame(
        rows, columns=["match_id", "competition_stage", "home_team", "away_team", "home_score", "away_score",
                       "shootout", "shootout_winner"],
    ).assign(tournament=tournament, match_date=pd.Timestamp("2025-11-03"), neutral_ground=True)
    matches.loc[matches.index[:2], "neutral_ground"] = False
    elo = pd.DataFrame({"tournament": tournament, "team": teams, "elo": [ratings[t] for t in teams]})
    return matches, elo


def test_run_writes_every_table_workbook_and_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    monkeypatch.setattr(xp, "LUCK_SIMS", 500)
    monkeypatch.setattr(xp, "PROGRESS_SIMS", 200)
    cfg = load_config(EXAMPLE)
    men, men_elo = _cup("men", n_groups=3, first_round=8, seed=3)
    women, women_elo = _cup("women", n_groups=2, first_round=4, seed=5)
    women["match_id"] += 1000
    tds = pd.DataFrame({"team": men_elo["team"], "best_finish_rank": np.arange(12) % 8 + 1,
                        "campaigns_incl_2025": np.arange(12) % 5 + 1})
    write_table(cfg, "matches", pd.concat([men, women], ignore_index=True))
    write_table(cfg, "elo_pre_tournament", pd.concat([men_elo, women_elo], ignore_index=True))
    write_table(cfg, "tds_team", tds)

    xp.run(cfg)

    summary = read_table(cfg, "team_summary")
    assert len(summary) == 12 + 8
    assert summary["band_low"].notna().all() and summary["expected_stage"].notna().all()
    for name in ("expected_results", "overperformance", "overperformance_luck", "luck_summary",
                 "expectation_calibration", "expected_progress", "overperformance_sensitivity",
                 "sensitivity_scores", "sensitivity_comparison"):
        assert len(read_table(cfg, name)) > 0, name
    checks = read_table(cfg, "checks_expectation")
    validity = checks[~checks["check"].str.contains("Spearman with main")]
    assert validity["passed"].all(), validity.loc[~validity["passed"], "check"].tolist()
    assert set(read_table(cfg, "sensitivity_comparison")["variant"]) >= {"host_100", "tds_campaigns"}

    sheets = pd.read_excel(cfg.build_dir / "expectation.xlsx", sheet_name=None)
    assert list(sheets)[0] == "team_summary" and "checks_expectation" in sheets
    figures = sorted(p.name for p in (cfg.build_dir / "figures").glob("*.html"))
    assert len(figures) == 8 and (cfg.build_dir / "figures" / "plotly.min.js").is_file()
    out = capsys.readouterr().out
    assert "=== teams beyond luck" in out and "=== women: bottom 10" in out
