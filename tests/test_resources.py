from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from three_trees.features import resources as res
from three_trees.features import spread as sp

A, B = 1, 2
NAMES = {A: "Alpha", B: "Beta"}


def ev(type_: str, team_id: int = A, period: int = 1, secs: float = 0.0, match_id: int = 100,
       **extra) -> dict:
    minute = (45 if period == 2 else 0) + int(secs // 60)
    return {
        "tournament": "men", "match_id": match_id, "period": period, "period_seconds": float(secs),
        "minute": minute, "second": int(secs % 60), "type": type_, "team_id": team_id, "team": NAMES[team_id],
        "player_id": np.nan, "position": None, "tactics": None, "substitution_replacement_id": np.nan,
        "foul_committed_card": None, "bad_behaviour_card": None, "player_off_permanent": False,
        "possession": 1, "possession_team_id": team_id, "play_pattern": "Regular Play",
        "possession_directness": np.nan, "location_x": 50.0, "location_y": 40.0,
        "pass_end_location_x": np.nan, "carry_end_location_x": np.nan, "pass_outcome": None,
        "shot_type": None, "shot_statsbomb_xg": np.nan, "shot_key_pass_id": None,
        "pass_assisted_shot_id": None,
        "pass_shot_assist": False, "pass_goal_assist": False, "related_events": None, "duel_type": None,
        **extra,
    }


def lineup(players: list[tuple[int, int, str]]) -> str:
    return str({"formation": 433, "lineup": [
        {"player": {"id": pid, "name": "x"}, "position": {"id": 1, "name": pos}, "jersey_number": num}
        for pid, num, pos in players]})


def frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df.sort_values(["match_id", "period", "period_seconds"], kind="stable").reset_index(drop=True)
    df["index"] = df.groupby("match_id").cumcount() + 1
    df["id"] = [f"e{i}" for i in range(len(df))] if "id" not in df else df["id"].fillna(
        pd.Series([f"e{i}" for i in range(len(df))]))
    return df


def matches(dates: dict[int, str] | None = None) -> pd.DataFrame:
    dates = dates or {100: "2025-11-03"}
    return pd.DataFrame([{
        "tournament": "men", "match_id": mid, "match_date": pd.Timestamp(d).date(), "kick_off": "15:00:00",
        "home_team": "Alpha", "away_team": "Beta", "home_team_id": A, "away_team_id": B,
        "home_score": 0, "away_score": 0,
    } for mid, d in dates.items()])


def minutes_match() -> pd.DataFrame:
    """Period 1 lasts 46 min, period 2 48 min. Player 12 sent off at 30:00; player 13 replaced by 14
    at 55:00."""
    rows = [
        ev("Starting XI", tactics=lineup([(11, 1, "Goalkeeper"), (12, 4, "Center Back"),
                                          (13, 9, "Center Forward")])),
        ev("Pass", secs=10, player_id=13.0, position="Center Forward"),
        ev("Foul Committed", secs=1800, player_id=12.0, foul_committed_card="Red Card"),
        ev("Half End", secs=2760),
        ev("Substitution", period=2, secs=600, player_id=13.0, position="Center Forward",
           substitution_replacement_id=14.0),
        ev("Tactical Shift", period=2, secs=610,
           tactics=lineup([(11, 1, "Goalkeeper"), (14, 20, "Left Wing")])),
        ev("Pass", period=2, secs=900, player_id=14.0, position="Left Wing"),
        ev("Half End", period=2, secs=2880),
    ]
    return frame(rows)


@pytest.fixture
def minutes() -> pd.DataFrame:
    return res.player_match_minutes(minutes_match()).set_index("player_id")


def test_minutes_with_substitution_and_red_card(minutes: pd.DataFrame) -> None:
    assert minutes.loc[11, "minutes"] == pytest.approx(94.0)
    assert minutes.loc[12, "minutes"] == pytest.approx(30.0)
    assert minutes.loc[12, "off_minute"] == pytest.approx(30.0)
    assert minutes.loc[12, "off_reason"] == "red card"
    assert minutes.loc[13, "minutes"] == pytest.approx(56.0)
    assert minutes.loc[13, "off_minute"] == pytest.approx(55.0)
    assert minutes.loc[14, "minutes"] == pytest.approx(38.0)
    assert minutes.loc[14, "on_minute"] == pytest.approx(55.0)
    assert np.isnan(minutes.loc[11, "off_minute"])
    assert minutes["started"].to_dict() == {11: True, 12: True, 13: True, 14: False}


def test_substitute_position_and_jersey_from_tactical_shift(minutes: pd.DataFrame) -> None:
    assert minutes.loc[14, "position"] == "Left Wing"
    assert minutes.loc[14, "position_group"] == "Forward"
    assert minutes.loc[14, "jersey_number"] == 20
    assert minutes.loc[12, "position_group"] == "Defender"


def test_permanent_player_off_ends_time() -> None:
    rows = [ev("Starting XI", tactics=lineup([(11, 1, "Goalkeeper")])),
            ev("Player Off", secs=600, player_id=11.0, player_off_permanent=True),
            ev("Half End", secs=2700), ev("Half End", period=2, secs=2700)]
    pm = res.player_match_minutes(frame(rows)).set_index("player_id")
    assert pm.loc[11, "minutes"] == pytest.approx(10.0)
    assert pm.loc[11, "off_reason"] == "player off"


@pytest.mark.parametrize("position, group", [
    ("Goalkeeper", "Goalkeeper"), ("Left Wing Back", "Defender"), ("Right Center Back", "Defender"),
    ("Center Defensive Midfield", "Midfielder"), ("Left Attacking Midfield", "Midfielder"),
    ("Right Wing", "Forward"), ("Secondary Striker", "Forward"), ("Center Forward", "Forward"), (None, None),
])
def test_position_group(position: str | None, group: str | None) -> None:
    assert res.position_group(position) == group


def two_matches() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Alpha starts 11, 12, 13 on 3 Nov, then 11, 12, 15 on 6 Nov (one change)."""
    rows = []
    for mid, xi in ((100, [11, 12, 13]), (101, [11, 12, 15])):
        rows += [ev("Starting XI", match_id=mid, tactics=lineup([(p, p, "Center Back") for p in xi])),
                 ev("Starting XI", B, match_id=mid, tactics=lineup([(21, 1, "Goalkeeper")])),
                 ev("Half End", match_id=mid, secs=2700), ev("Half End", match_id=mid, period=2, secs=2700)]
    return frame(rows), matches({100: "2025-11-03", 101: "2025-11-06"})


def test_context_and_starters_changed() -> None:
    events, m = two_matches()
    pm = res.player_match_minutes(events)
    context = res.match_context(m)
    alpha = context[context["team_id"] == A].set_index("match_id")
    assert alpha["match_number"].to_dict() == {100: 1, 101: 2}
    assert np.isnan(alpha.loc[100, "rest_days"]) and alpha.loc[101, "rest_days"] == 3

    changed = sp.starters_changed(pm, context).query("team_id == @A").set_index("match_id")
    changed = changed["starters_changed"]
    assert np.isnan(changed[100]) and changed[101] == 1

    load = res.squad_load(pm, context).query("team_id == @A").set_index("match_id")
    assert load.loc[100, "starters_minutes_cum"] == 0
    assert load.loc[101, "starters_minutes_cum"] == pytest.approx(2 * 90.0)
    assert load.loc[101, "players_used"] == 3


def test_minutes_outside_top11() -> None:
    rows = [{"tournament": "men", "match_id": 100, "team_id": A, "player_id": p, "minutes": 90.0 - p}
            for p in range(12)]
    out = sp.minutes_outside_top11(pd.DataFrame(rows)).iloc[0]
    total = sum(90.0 - p for p in range(12))
    assert out["minutes_outside_top11_share"] == pytest.approx((90.0 - 11) / total)


def chance_match() -> pd.DataFrame:
    rows = [
        ev("Pass", A, id="p1", player_id=7.0, location_x=70.0),
        ev("Shot", A, secs=5, id="s1", player_id=9.0, shot_type="Open Play", shot_statsbomb_xg=0.3,
           shot_key_pass_id="p1", location_x=105.0),
        ev("Pass", A, secs=20, id="p2", player_id=8.0, pass_shot_assist=True, related_events="['s2']",
           location_x=90.0, location_y=10.0),
        ev("Shot", A, secs=25, id="s2", player_id=9.0, shot_type="Open Play", shot_statsbomb_xg=0.2,
           location_x=110.0),
        ev("Pass", A, secs=40, id="p3", player_id=7.0, shot_type=None),
        ev("Shot", A, secs=45, id="s3", player_id=9.0, shot_type="Penalty", shot_statsbomb_xg=0.78,
           shot_key_pass_id="p3", location_x=108.0),
        ev("Shot", B, secs=60, id="s4", player_id=21.0, shot_type="Open Play", shot_statsbomb_xg=0.1,
           location_x=100.0),
    ]
    return frame(rows)


def test_xa_is_credited_to_the_passer() -> None:
    credits = res.shot_credits(chance_match())
    assert set(credits["shot_id"]) == {"s1", "s2", "s4"}
    threat = sp.player_threat(credits).set_index("player_id")
    assert threat.loc[7.0, "xa"] == pytest.approx(0.3)
    assert threat.loc[8.0, "xa"] == pytest.approx(0.2)
    assert threat.loc[9.0, "xg"] == pytest.approx(0.5)


def test_chances_table() -> None:
    tm = pd.DataFrame({"tournament": "men", "match_id": 100, "team_id": [A, B], "opponent_id": [B, A],
                       "np_xg": [0.5, 0.1], "territory_share": [0.8, 0.0]})
    out = res.chances(chance_match(), tm).set_index("team_id")
    assert out.loc[A, "xa"] == pytest.approx(0.5)
    assert out.loc[B, "xa"] == 0
    assert out.loc[A, "np_xg_against"] == pytest.approx(0.1)
    assert out.loc[A, "conversion"] == pytest.approx(0.5 / 0.8)
    assert np.isnan(out.loc[B, "conversion"])


def test_final_third_entries_and_security() -> None:
    rows = [
        ev("Pass", B, location_x=70.0, pass_end_location_x=85.0),
        ev("Pass", B, location_x=70.0, pass_end_location_x=85.0, pass_outcome="Incomplete"),
        ev("Carry", B, location_x=79.0, carry_end_location_x=81.0),
        ev("Carry", B, location_x=81.0, carry_end_location_x=90.0),
        ev("Shot", B, id="s1", shot_type="Open Play", shot_statsbomb_xg=0.4),
    ]
    tm = pd.DataFrame({"tournament": "men", "match_id": 100, "team_id": [A, B], "opponent_id": [B, A],
                       "np_xg": [0.0, 0.4], "territory_share": [0.5, 0.5]})
    out = res.chances(frame(rows), tm).set_index("team_id")
    assert out.loc[A, "opponent_final_third_entries"] == 2
    assert out.loc[A, "security"] == pytest.approx(0.2)
    assert np.isnan(out.loc[B, "security"])


def test_defensive_work_and_ppda() -> None:
    rows = [
        ev("Pressure", A, location_x=30.0), ev("Pressure", A, location_x=90.0),
        ev("Duel", A, duel_type="Tackle", location_x=60.0), ev("Duel", A, duel_type="Aerial Lost"),
        ev("Interception", A, location_x=40.0), ev("Foul Committed", A, location_x=70.0),
        ev("Block", A, location_x=20.0),
        ev("Pass", B, location_x=30.0), ev("Pass", B, location_x=60.0), ev("Pass", B, location_x=71.0),
        ev("Pass", B, location_x=80.0),
    ]
    tm = pd.DataFrame({"tournament": "men", "match_id": 100, "team_id": [A, B], "opponent_id": [B, A],
                       "dominance": [0.5, 0.5]})
    out = res.defensive_work(frame(rows), tm).set_index("team_id")
    assert out.loc[A, "pressures"] == 2
    assert out.loc[A, "defensive_actions"] == 6
    assert out.loc[A, "defensive_work_per_dominance"] == pytest.approx(6 / 50)
    assert out.loc[A, "ppda"] == pytest.approx(3 / 2)
    assert np.isnan(out.loc[B, "ppda"])


def test_possession_time_and_shares() -> None:
    rows = [
        ev("Starting XI", A, possession=1),
        ev("Pass", A, secs=0, possession=1, possession_directness=0.4),
        ev("Pass", A, secs=30, possession=1, possession_directness=0.4),
        ev("Pass", B, secs=40, possession=2, possession_team_id=B),
        ev("Pressure", A, secs=70, possession=2, possession_team_id=B),
        ev("Pass", B, secs=100, possession=2, possession_team_id=B, possession_directness=0.8),
        ev("Half End", B, secs=2000, possession=2, possession_team_id=B),
        ev("Pass", A, period=2, secs=0, possession=2, possession_team_id=B),
        ev("Pass", A, period=2, secs=30, possession=3),
    ]
    events = frame(rows)
    poss = res.possessions(events)
    assert len(poss) == 4
    tm = pd.DataFrame({"tournament": "men", "match_id": 100, "team_id": [A, B], "opponent_id": [B, A],
                       "pressures": [1, 0]})
    out = res.possession_measures(events, tm).set_index("team_id")
    assert out.loc[A, "possession_minutes"] == pytest.approx(0.5)
    assert out.loc[B, "possession_minutes"] == pytest.approx(1.0)
    assert out.loc[A, "possession_share"] + out.loc[B, "possession_share"] == pytest.approx(1.0)
    assert out.loc[A, "possession_share"] == pytest.approx(1 / 3)
    assert out.loc[A, "pressures_per_opp_possession_min"] == pytest.approx(1.0)
    assert out.loc[A, "directness"] == pytest.approx(0.4)
    assert out.loc[B, "directness"] == pytest.approx(0.8)
