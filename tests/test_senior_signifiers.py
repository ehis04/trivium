from __future__ import annotations

import dataclasses
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trivium import synthetic
from trivium.config import load_config
from trivium.ingest import open_data as od
from trivium.ingest import statsbomb as sb
from trivium.models import senior_signifiers as ss
from trivium.storage import write_table

EXAMPLE_CONFIG = Path(__file__).resolve().parents[1] / "config.example.yaml"


def _make_senior_tournament(
    rng: np.random.Generator,
    year: int = 2024,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Synthetic senior tournament (32 teams, 64 matches) with prior friendlies and events."""
    matches, _ = synthetic.demo_matches("men", 8, 16, rng)
    offset = date(year, 6, 1) - matches["match_date"].min()
    matches["match_date"] = matches["match_date"] + offset
    matches["tournament"] = "senior_men_test"

    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    friendlies = []
    for k in range(350):
        a, b = rng.choice(teams, 2, replace=False)
        friendlies.append(
            {
                "date": date(year - 1, 1, 1) + timedelta(days=k % 360),
                "home_team": a,
                "away_team": b,
                "home_score": int(rng.poisson(1.3)),
                "away_score": int(rng.poisson(1.1)),
                "tournament": "Friendly",
                "city": "x",
                "country": "y",
                "neutral": False,
            }
        )
    results = pd.DataFrame(friendlies)

    # Generate events for each match
    events_list: list[dict[str, object]] = []
    for _, m in matches.iterrows():
        mid = int(m["match_id"])
        hid, aid = int(m["home_team_id"]), int(m["away_team_id"])
        ht, at = str(m["home_team"]), str(m["away_team"])

        if mid == 1:
            # Explicit hand-counted actions for match 1:
            # Home: 3 final-third actions (x >= 80), Away: 1 final-third action
            for i in range(3):
                events_list.append(
                    {
                        "tournament": "senior_men_test",
                        "match_id": mid,
                        "period": 1,
                        "type": "Pass",
                        "team_id": hid,
                        "team": ht,
                        "location_x": 85.0 + i,
                        "location_y": 40.0,
                        "location_bucket": None,
                        "duel_type": None,
                        "duel_outcome": None,
                        "50_50": None,
                        "related_events": None,
                        "ball_recovery_recovery_failure": False,
                        "pass_aerial_won": False,
                        "shot_type": None,
                        "shot_statsbomb_xg": np.nan,
                        "id": f"e_{mid}_h_{i}",
                    }
                )
            events_list.append(
                {
                    "tournament": "senior_men_test",
                    "match_id": mid,
                    "period": 1,
                    "type": "Pass",
                    "team_id": aid,
                    "team": at,
                    "location_x": 90.0,
                    "location_y": 40.0,
                    "location_bucket": None,
                    "duel_type": None,
                    "duel_outcome": None,
                    "50_50": None,
                    "related_events": None,
                    "ball_recovery_recovery_failure": False,
                    "pass_aerial_won": False,
                    "shot_type": None,
                    "shot_statsbomb_xg": np.nan,
                    "id": f"e_{mid}_a_0",
                }
            )
            # Away also has a pass in its own third (x = 30)
            events_list.append(
                {
                    "tournament": "senior_men_test",
                    "match_id": mid,
                    "period": 1,
                    "type": "Pass",
                    "team_id": aid,
                    "team": at,
                    "location_x": 30.0,
                    "location_y": 40.0,
                    "location_bucket": None,
                    "duel_type": None,
                    "duel_outcome": None,
                    "50_50": None,
                    "related_events": None,
                    "ball_recovery_recovery_failure": False,
                    "pass_aerial_won": False,
                    "shot_type": None,
                    "shot_statsbomb_xg": np.nan,
                    "id": f"e_{mid}_a_def",
                }
            )
        else:
            # Other matches: 2 actions for each side
            for side_id, side_team in ((hid, ht), (aid, at)):
                events_list.append(
                    {
                        "tournament": "senior_men_test",
                        "match_id": mid,
                        "period": 1,
                        "type": "Pass",
                        "team_id": side_id,
                        "team": side_team,
                        "location_x": 82.0,
                        "location_y": 40.0,
                        "location_bucket": None,
                        "duel_type": None,
                        "duel_outcome": None,
                        "50_50": None,
                        "related_events": None,
                        "ball_recovery_recovery_failure": False,
                        "pass_aerial_won": False,
                        "shot_type": None,
                        "shot_statsbomb_xg": np.nan,
                        "id": f"e_{mid}_{side_id}_pass",
                    }
                )

        # Tackles
        events_list.append(
            {
                "tournament": "senior_men_test",
                "match_id": mid,
                "period": 1,
                "type": "Duel",
                "team_id": hid,
                "team": ht,
                "location_x": 50.0,
                "location_y": 40.0,
                "location_bucket": None,
                "duel_type": "Tackle",
                "duel_outcome": "Won",
                "50_50": None,
                "related_events": None,
                "ball_recovery_recovery_failure": False,
                "pass_aerial_won": False,
                "shot_type": None,
                "shot_statsbomb_xg": np.nan,
                "id": f"e_{mid}_tackle_h",
            }
        )
        events_list.append(
            {
                "tournament": "senior_men_test",
                "match_id": mid,
                "period": 1,
                "type": "Duel",
                "team_id": aid,
                "team": at,
                "location_x": 50.0,
                "location_y": 40.0,
                "location_bucket": None,
                "duel_type": "Tackle",
                "duel_outcome": "Lost In Play",
                "50_50": None,
                "related_events": None,
                "ball_recovery_recovery_failure": False,
                "pass_aerial_won": False,
                "shot_type": None,
                "shot_statsbomb_xg": np.nan,
                "id": f"e_{mid}_tackle_a",
            }
        )
        # Shots
        events_list.append(
            {
                "tournament": "senior_men_test",
                "match_id": mid,
                "period": 1,
                "type": "Shot",
                "team_id": hid,
                "team": ht,
                "location_x": 105.0,
                "location_y": 40.0,
                "location_bucket": None,
                "duel_type": None,
                "duel_outcome": None,
                "50_50": None,
                "related_events": None,
                "ball_recovery_recovery_failure": False,
                "pass_aerial_won": False,
                "shot_type": "Open Play",
                "shot_statsbomb_xg": 0.25,
                "id": f"e_{mid}_shot_h",
            }
        )

    return matches, results, _adapter_shape(pd.DataFrame(events_list))


def _adapter_shape(events: pd.DataFrame) -> pd.DataFrame:
    """Every column of the open-data adapter's senior_events; one possession per event."""
    events = events.reindex(columns=list(dict.fromkeys([*events.columns, *od.SENIOR_EVENT_COLUMNS])))
    order = events.groupby("match_id").cumcount()
    flags = {c: sb.to_flag(events[c]) for c in sb.FLAG_COLUMNS if c in events}
    return events.assign(**flags, index=order + 1, possession=order + 1, possession_team_id=events["team_id"],
                         period_seconds=10.0 * order, minute=0, second=10 * order,
                         play_pattern="Regular Play")


def test_field_tilt_matches_hand_count() -> None:
    rng = np.random.default_rng(ss.SEED)
    matches, _, events = _make_senior_tournament(rng)
    tm = ss.senior_team_match(events, matches)

    m1 = tm[tm["match_id"] == 1]
    assert len(m1) == 2
    h_row = m1[m1["team"] == matches.loc[0, "home_team"]].iloc[0]
    a_row = m1[m1["team"] == matches.loc[0, "away_team"]].iloc[0]

    # Home has 3 passes in final third (x >= 80), away has 1
    # territory_share = 3 / (3 + 1) = 0.75 for home, 0.25 for away
    assert h_row["territory_share"] == pytest.approx(0.75)
    assert a_row["territory_share"] == pytest.approx(0.25)


def test_senior_expectation_probabilities() -> None:
    rng = np.random.default_rng(ss.SEED)
    matches, results, _ = _make_senior_tournament(rng)
    op = ss.senior_expectation(matches, results)

    assert len(op) == len(matches) * 2
    assert set(op.columns).issuperset(
        {"p_loss", "p_draw", "p_win", "expected_points", "actual_points", "overperformance"}
    )
    probs_sum = op["p_loss"] + op["p_draw"] + op["p_win"]
    assert np.allclose(probs_sum, 1.0, atol=1e-6)
    assert (op["expected_points"] >= 0).all()
    assert (op["expected_points"] <= 3).all()


def test_repeatable_measure_and_noise() -> None:
    rng = np.random.default_rng(ss.SEED)
    matches, results, events = _make_senior_tournament(rng)
    tm = ss.senior_team_match(events, matches)
    op = ss.senior_expectation(matches, results)

    # 1. Repeatable signifier (conversion): strong team fixed effect + small noise
    tm["conversion"] = tm["team_id"] * 5.0 + rng.normal(0, 0.1, len(tm))
    # 2. Pure-noise signifier (security): N(0, 1)
    tm["security"] = rng.normal(0, 1.0, len(tm))

    sig = ss.senior_signifiers(tm, op, matches, n_boot=200, seed=ss.SEED)

    rep = sig[sig["measure"] == "conversion"].iloc[0]
    noise = sig[sig["measure"] == "security"].iloc[0]

    assert rep["r_sb"] >= 0.5
    assert bool(rep["passed_reliability"]) is True
    assert noise["r_sb"] < 0.5
    assert bool(noise["passed_reliability"]) is False


def test_measure_tracking_overperformance_excludes_zero() -> None:
    rng = np.random.default_rng(ss.SEED)
    matches, results, events = _make_senior_tournament(rng)
    tm = ss.senior_team_match(events, matches)
    op = ss.senior_expectation(matches, results)

    merged = tm.merge(op[["match_id", "team", "overperformance"]], on=["match_id", "team"])
    # Threat spread constructed to correlate strongly with over-performance
    tm["top2_xg_xa_share"] = merged["overperformance"] * 3.0 + rng.normal(0, 0.05, len(tm))

    sig = ss.senior_signifiers(tm, op, matches, n_boot=300, seed=ss.SEED)
    track_row = sig[sig["measure"] == "top2_xg_xa_share"].iloc[0]

    assert track_row["r_overperformance"] > 0.5
    assert track_row["r_low"] > 0.0
    assert bool(track_row["excludes_zero"]) is True


def test_missing_senior_events_returns_empty(tmp_path: Path) -> None:
    cfg = load_config(EXAMPLE_CONFIG)
    cfg = dataclasses.replace(cfg, build_dir=tmp_path)
    res = ss.run_senior_signifiers(cfg)
    assert res == {}


def test_team_with_no_elo_history_raises_naming_team() -> None:
    rng = np.random.default_rng(ss.SEED)
    matches, results, _ = _make_senior_tournament(rng)

    # Remove all friendlies for Team A1
    involved = (results["home_team"] == "Team A1") | (results["away_team"] == "Team A1")
    reduced_results = results[~involved].copy()

    with pytest.raises(ValueError, match="Team A1"):
        ss.senior_expectation(matches, reduced_results)


def test_checks_senior_signifiers_schema() -> None:
    table = pd.DataFrame(
        [
            {
                "tournament": "senior_men_test",
                "measure": "territory_share",
                "label": "Field tilt",
                "signifier": 1,
                "n_teams": 32,
                "n_rows": 64,
                "r_sb": 0.65,
                "passed_reliability": True,
                "r_overperformance": 0.35,
                "r_low": 0.10,
                "r_high": 0.55,
                "excludes_zero": True,
                "corr_with_elo": 0.40,
            }
        ]
    )
    checks = ss.checks_senior_signifiers(table)

    assert len(checks) == 2
    assert list(checks.columns) == ["check", "target", "value", "passed"]
    rel_check = checks.iloc[0]
    op_check = checks.iloc[1]

    assert "repeatable" in rel_check["check"]
    assert rel_check["target"] == 0.5
    assert rel_check["value"] == 0.65
    assert bool(rel_check["passed"]) is True

    assert "tracks over-performance" in op_check["check"]
    assert np.isnan(op_check["target"])
    assert op_check["value"] == 0.35
    assert bool(op_check["passed"]) is True


def test_run_senior_signifiers_full_pipeline(tmp_path: Path) -> None:
    rng = np.random.default_rng(ss.SEED)
    matches, results, events = _make_senior_tournament(rng)

    cfg = load_config(EXAMPLE_CONFIG)
    cfg = dataclasses.replace(cfg, build_dir=tmp_path)

    write_table(cfg, "senior_events", events)
    write_table(cfg, "senior_matches", matches)

    out = ss.run_senior_signifiers(
        cfg, results={"men": results, "women": results}, n_boot=200
    )
    expected_tables = {
        "senior_team_match",
        "senior_overperformance",
        "senior_signifiers",
        "checks_senior_signifiers",
    }
    assert set(out.keys()) == expected_tables

    for name in expected_tables:
        parquet_path = cfg.build_dir / f"{name}.parquet"
        assert parquet_path.is_file(), f"{name}.parquet was not written"
        assert not out[name].empty
