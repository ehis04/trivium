from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from three_trees import synthetic
from three_trees.config import DATA_ROOT_ENV, load_config
from three_trees.models import senior

EXAMPLE = Path(__file__).resolve().parents[1] / "config.example.yaml"


def _results(year: int = 2030) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A fake senior World Cup (8 groups, round of 16) in martj42 format, with prior friendlies."""
    rng = np.random.default_rng(4)
    matches, _ = synthetic.demo_matches("men", 8, 16, rng)
    offset = date(year, 6, 1) - matches["match_date"].min()
    wc = pd.DataFrame({
        "date": matches["match_date"] + offset,
        "home_team": matches["home_team"], "away_team": matches["away_team"],
        "home_score": matches["home_score"], "away_score": matches["away_score"],
        "tournament": "FIFA World Cup", "city": "x", "country": "y", "neutral": matches["neutral_ground"],
    })
    teams = sorted(set(wc["home_team"]) | set(wc["away_team"]))
    friendlies = []
    for k in range(300):
        a, b = rng.choice(teams, 2, replace=False)
        friendlies.append({"date": date(year - 1, 1, 1) + timedelta(days=k), "home_team": a, "away_team": b,
                           "home_score": int(rng.poisson(1.3)), "away_score": int(rng.poisson(1.1)),
                           "tournament": "Friendly", "city": "x", "country": "y", "neutral": False})
    results = pd.concat([pd.DataFrame(friendlies), wc], ignore_index=True)
    so = matches[matches["shootout"]]
    shootouts = pd.DataFrame({"date": so["match_date"] + offset, "home_team": so["home_team"],
                              "away_team": so["away_team"], "winner": so["shootout_winner"]})
    return results, shootouts


def test_senior_world_cup_shapes() -> None:
    results, shootouts = _results()
    matches, elo_pre = senior.senior_world_cup(results, shootouts, "men", 2030)
    assert len(matches) == 64
    assert matches["competition_stage"].value_counts()["Group Stage"] == 48
    assert matches["shootout"].sum() == len(shootouts)
    assert (matches.loc[matches["shootout"], "shootout_winner"] != "").all()
    assert len(elo_pre) == 32 and elo_pre["elo"].notna().all()


def test_unsupported_format_raises() -> None:
    results, shootouts = _results()
    with pytest.raises(ValueError, match="16-team knockout"):
        senior.senior_world_cup(results.drop(index=results.index[-1]), shootouts, "men", 2030)


def test_team_without_elo_history_is_named() -> None:
    results, shootouts = _results()
    friendlies = results["tournament"] == "Friendly"
    involved = (results["home_team"] == "Team A1") | (results["away_team"] == "Team A1")
    with pytest.raises(ValueError, match="Team A1"):
        senior.senior_world_cup(results[~(friendlies & involved)], shootouts, "men", 2030)


def test_backtest_edition_metrics() -> None:
    results, shootouts = _results()
    matches, elo_pre = senior.senior_world_cup(results, shootouts, "men", 2030)
    out = senior.backtest_edition(matches, elo_pre, luck_sims=300)
    assert out["teams"] == 32 and out["matches"] == 64
    assert all(np.isfinite(value) for value in out.values())
    assert out["model_beats_base_rates"] in (0.0, 1.0)
    checks = senior.checks_senior(pd.DataFrame([{"tournament": "men", "year": 2030, **out}]))
    assert len(checks) == 4 and set(checks.columns) == {"check", "target", "value", "passed"}


def test_download_shootouts_keeps_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    cfg = load_config(EXAMPLE)
    cfg.external_dir.mkdir(parents=True)
    for t in ("men", "women"):
        senior.shootouts_path(cfg, t).write_text("date,home_team,away_team,winner\n", encoding="utf-8")

    def no_network(*args, **kwargs):
        raise AssertionError("download attempted")

    monkeypatch.setattr(senior.urllib.request, "urlopen", no_network)
    assert set(senior.download_shootouts(cfg)) == {"men", "women"}


def test_run_builds_open_data_only_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from three_trees.ingest import open_data
    from three_trees.models import senior_signifiers
    from three_trees.storage import write_table

    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    cfg = load_config(EXAMPLE)
    results, shootouts = _results()
    monkeypatch.setattr(senior.ex, "download_results", lambda c: {"men": "m", "women": "w"})
    monkeypatch.setattr(senior, "download_shootouts", lambda c: {"men": "m", "women": "w"})
    monkeypatch.setattr(senior.ex, "load_results", lambda p: results)
    monkeypatch.setattr(senior, "load_shootouts", lambda p: shootouts)
    backtest = senior.backtest
    monkeypatch.setattr(senior, "backtest", lambda r, s: backtest(
        {"men": results}, {"men": shootouts}, {"men": (2030,)}, luck_sims=50))
    adapter_runs, signifier_runs = [], []

    def adapter(c) -> None:
        adapter_runs.append(1)
        write_table(c, "senior_events", pd.DataFrame({"x": [1]}))
        write_table(c, "senior_matches", pd.DataFrame({"x": [1]}))

    monkeypatch.setattr(open_data, "run", adapter)
    monkeypatch.setattr(senior_signifiers, "run_senior_signifiers", lambda c, results: signifier_runs.append(1))
    senior.run(cfg)
    senior.run(cfg)
    assert adapter_runs == [1] and signifier_runs == [1, 1]
