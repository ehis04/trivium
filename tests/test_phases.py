from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trivium.features import phases as ph

A, B = 1, 2


def first_events(rows: list[tuple]) -> pd.DataFrame:
    """rows: (play_pattern, is_transition, control_degree, is_controlled_possession)."""
    cols = ["play_pattern", "is_transition", "control_degree", "is_controlled_possession"]
    return pd.DataFrame(rows, columns=cols)


def test_phase_priority() -> None:
    df = first_events([
        ("From Corner", True, 0.0, True),
        ("From Throw In", False, 1.0, True),
        ("From Counter", True, 0.0, False),
        ("Regular Play", True, 0.0, True),
        ("Regular Play", False, 1.0, False),
        ("From Keeper", False, np.nan, True),
        ("From Goal Kick", False, 0.0, False),
    ])
    assert ph.phase_of(df).tolist() == [
        "set_piece", "set_piece", "counter_attack", "transition", "settled", "settled", "other"]


def test_phase_without_transition_fields() -> None:
    df = pd.DataFrame({"play_pattern": ["From Free Kick", "Regular Play", "From Counter"]})
    assert ph.phase_of(df).tolist() == ["set_piece", "other", "counter_attack"]


def ev(type_: str, team_id: int, secs: float, possession: int, **extra) -> dict:
    return {
        "tournament": "men", "match_id": 100, "period": 1, "period_seconds": secs, "type": type_,
        "team_id": team_id, "possession": possession, "possession_team_id": extra.pop("poss_team", team_id),
        "play_pattern": "Regular Play", "is_transition": False, "control_degree": 1.0,
        "is_controlled_possession": True, "shot_type": None, "shot_statsbomb_xg": np.nan,
        "obv_total_net": np.nan, **extra,
    }


def phase_events() -> pd.DataFrame:
    rows = [
        ev("Starting XI", A, 0, 1, play_pattern="From Kick Off", control_degree=np.nan),
        ev("Pass", A, 0, 1, obv_total_net=0.01),
        ev("Shot", A, 20, 1, shot_type="Open Play", shot_statsbomb_xg=0.3, obv_total_net=0.05),
        ev("Pass", B, 30, 2, play_pattern="From Corner"),
        ev("Pressure", A, 35, 2, poss_team=B),
        ev("Shot", B, 40, 2, shot_type="Open Play", shot_statsbomb_xg=0.1, play_pattern="From Corner"),
        ev("Ball Recovery", A, 50, 3, is_transition=True, control_degree=0.0, is_controlled_possession=False),
        ev("Shot", A, 55, 3, shot_type="Penalty", shot_statsbomb_xg=0.78),
    ]
    df = pd.DataFrame(rows)
    df["index"] = range(1, len(df) + 1)
    df["id"] = [f"e{i}" for i in range(len(df))]
    return df


def matches() -> pd.DataFrame:
    return pd.DataFrame([{"tournament": "men", "match_id": 100, "home_team": "Alpha", "away_team": "Beta",
                          "home_team_id": A, "away_team_id": B}])


def test_team_match_phase() -> None:
    out = ph.team_match_phase(phase_events(), matches()).set_index(["team", "phase"])
    assert len(out) == 2 * len(ph.PHASES)
    assert out.loc[("Alpha", "settled"), "possessions"] == 1
    assert out.loc[("Alpha", "settled"), "possession_seconds"] == 20
    assert out.loc[("Alpha", "settled"), "xg"] == pytest.approx(0.3)
    assert out.loc[("Alpha", "settled"), "obv"] == pytest.approx(0.06)
    assert out.loc[("Alpha", "transition"), "shots"] == 0
    assert out.loc[("Alpha", "transition"), "share_of_possessions"] == pytest.approx(0.5)
    assert out.loc[("Beta", "set_piece"), "xg_per_possession"] == pytest.approx(0.1)
    assert np.isnan(out.loc[("Beta", "settled"), "xg_per_possession"])
    assert out.groupby(level="team")["share_of_possessions"].sum().tolist() == pytest.approx([1.0, 1.0])


def test_obv_missing_without_column() -> None:
    out = ph.team_match_phase(phase_events().drop(columns="obv_total_net"), matches())
    assert out["obv"].isna().all()


def test_team_tournament_phase_recomputes_ratios() -> None:
    one = ph.team_match_phase(phase_events(), matches())
    two = pd.concat([one, one.assign(match_id=101, xg=one["xg"] * 3)], ignore_index=True)
    out = ph.team_tournament_phase(two).set_index(["team", "phase"])
    assert out.loc[("Alpha", "settled"), "matches"] == 2
    assert out.loc[("Alpha", "settled"), "possessions"] == 2
    assert out.loc[("Alpha", "settled"), "xg_per_possession"] == pytest.approx((0.3 + 0.9) / 2)
    assert out.loc[("Alpha", "settled"), "share_of_possessions"] == pytest.approx(0.5)
