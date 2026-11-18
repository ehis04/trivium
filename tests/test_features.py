from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trivium.features import dominance as dom
from trivium.features import tables

A, B = 1, 2


def _event(type_: str, team_id: int, x: float = 50.0, y: float = 40.0, **extra) -> dict:
    return {
        "tournament": "men", "match_id": 100, "period": 1, "type": type_, "team_id": team_id,
        "team": "Alpha" if team_id == A else "Beta", "location_x": x, "location_y": y,
        "location_bucket": None, "duel_type": None, "duel_outcome": None, "50_50": None,
        "related_events": None, "ball_recovery_recovery_failure": False, "pass_aerial_won": False,
        "shot_type": None, "shot_statsbomb_xg": np.nan, "duel_duel_win_probability": np.nan,
        "pass_duel_win_probability": np.nan, "minute": 0, "second": 0, "possession": 1,
        "possession_team_id": team_id, "play_pattern": "Regular Play", "player_id": np.nan, "position": None,
        "tactics": None, "substitution_replacement_id": np.nan, "foul_committed_card": None,
        "bad_behaviour_card": None, "pass_end_location_x": np.nan, "carry_end_location_x": np.nan,
        "pass_outcome": None, "shot_key_pass_id": None, "pass_assisted_shot_id": None, **extra,
    }


def _matches() -> pd.DataFrame:
    return pd.DataFrame([{
        "tournament": "men", "match_id": 100, "home_team": "Alpha", "away_team": "Beta",
        "home_team_id": A, "away_team_id": B, "home_score": 1, "away_score": 0,
        "match_date": pd.Timestamp("2025-11-03").date(), "kick_off": "15:00:00",
    }])


def _match_events() -> pd.DataFrame:
    rows = [
        # Contests: A wins a tackle, B loses one (credited to A), B wins a tackle.
        _event("Duel", A, duel_type="Tackle", duel_outcome="Won"),
        _event("Duel", B, duel_type="Tackle", duel_outcome="Lost In Play"),
        _event("Duel", B, duel_type="Tackle", duel_outcome="Success In Play"),
        # One aerial won by A: B's Aerial Lost plus A's flagged pass (not counted again).
        _event("Duel", B, duel_type="Aerial Lost", duel_duel_win_probability=0.3),
        _event("Pass", A, x=30, pass_aerial_won=True, pass_duel_win_probability=0.7),
        # One 50/50, recorded once per player: A 'Lost', B 'Won' -> one contest for B.
        _event("50/50", A, id="f1", related_events="['f2']", **{"50_50": "{'outcome': {'id': 1, 'name': 'Lost'}}"}),
        _event("50/50", B, id="f2", related_events="['f1']", **{"50_50": "{'outcome': {'id': 4, 'name': 'Won'}}"}),
        # Second balls: A recovers once; B's failed recovery does not count.
        _event("Ball Recovery", A),
        _event("Ball Recovery", B, ball_recovery_recovery_failure=True),
        # Territory: A 3 actions in the final third, B 1 (its pass at x = 10 is in its own third).
        _event("Pass", A, x=85), _event("Carry", A, x=90), _event("Pass", A, x=105),
        _event("Pass", B, x=95), _event("Pass", B, x=10, y=10),
        # Chances: A 0.3 non-penalty xG (penalty excluded), B 0.1.
        _event("Shot", A, x=110, shot_type="Open Play", shot_statsbomb_xg=0.3),
        _event("Shot", A, x=108, shot_type="Penalty", shot_statsbomb_xg=0.78),
        _event("Shot", B, x=100, shot_type="Open Play", shot_statsbomb_xg=0.1),
    ]
    ev = pd.DataFrame(rows)
    ev["id"] = ev["id"].fillna(pd.Series([f"e{i}" for i in range(len(ev))]))
    return ev


@pytest.fixture
def team_match() -> pd.DataFrame:
    return dom.dominance_team_match(_match_events(), _matches()).set_index("team")


def test_known_shares(team_match: pd.DataFrame) -> None:
    a = team_match.loc["Alpha"]
    assert (a["duels_won"], a["aerials_won"], a["fifty_fifties_won"], a["second_balls_won"]) == (2, 1, 0, 1)
    assert a["contests_won"] == 4 and team_match.loc["Beta", "contests_won"] == 2
    assert a["contests_share"] == pytest.approx(4 / 6)
    assert a["territory_share"] == pytest.approx(3 / 4)
    assert a["np_xg"] == pytest.approx(0.3)
    assert a["chances_share"] == pytest.approx(0.75)
    assert a["dominance"] == pytest.approx((4 / 6 + 3 / 4 + 0.75) / 3)


def test_aerial_counted_once(team_match: pd.DataFrame) -> None:
    assert team_match["aerials_won"].sum() == 1
    assert team_match.loc["Alpha", "aerials_share"] == 1.0


def test_fifty_fifty_counted_once(team_match: pd.DataFrame) -> None:
    assert team_match["fifty_fifties_won"].to_dict() == {"Alpha": 0, "Beta": 1}


def test_shares_of_the_two_teams_sum_to_one(team_match: pd.DataFrame) -> None:
    for col in ("duels_share", "aerials_share", "fifty_fifties_share", "second_balls_share",
                "contests_share", "territory_share", "chances_share", "dominance"):
        assert team_match[col].sum() == pytest.approx(1.0)


def test_both_zero_share_is_half_and_flagged() -> None:
    s, flag = dom.share(pd.Series([0, 2]), pd.Series([0, 6]))
    assert s.tolist() == [0.5, 0.25]
    assert flag.tolist() == [True, False]


def test_zone_numbering() -> None:
    zones = dom.zone_of(pd.Series([0.0, 119.9, 120.0, 10.0]), pd.Series([0.0, 79.9, 80.0, 30.0]))
    assert zones.tolist() == [1, 18, 18, 2]


def test_opponent_action_is_mirrored_into_the_team_zone() -> None:
    zm = dom.team_match_zone(_match_events(), _matches())
    alpha = zm[zm["team"] == "Alpha"].set_index("zone")
    # Beta's pass at (10, 10) is at (110, 70) from Alpha's view: zone_x 5, zone_y 2 -> zone 18.
    assert alpha.loc[18, "zone_x"] == 5
    assert alpha.loc[18, "opponent_actions"] == 1
    assert alpha["opponent_actions"].sum() == 2
    assert len(zm) == 2 * dom.N_ZONES


def test_zone_shares_of_the_two_teams_sum_to_one() -> None:
    zm = dom.team_match_zone(_match_events(), _matches())
    a = zm[zm["team"] == "Alpha"].set_index("zone")
    b = zm[zm["team"] == "Beta"].set_index("zone")
    b.index = dom.N_ZONES + 1 - b.index
    used = a["actions_count"] > 0
    assert np.allclose(a.loc[used, "territory_share"] + b.loc[a.index[used], "territory_share"], 1.0)
    assert np.allclose(a.loc[used, "contests_share"] + b.loc[a.index[used], "contests_share"], 1.0)


def test_tournament_zone_sums_matches() -> None:
    zm = dom.team_match_zone(_match_events(), _matches())
    zt = dom.team_tournament_zone(pd.concat([zm, zm.assign(match_id=101)]))
    a = zt[zt["team"] == "Alpha"]
    assert a["matches"].eq(2).all()
    assert a["team_actions"].sum() == 2 * zm.loc[zm["team"] == "Alpha", "team_actions"].sum()


def _lineup(first_id: int) -> str:
    return str({"formation": 442, "lineup": [
        {"player": {"id": first_id + i, "name": "x"}, "position": {"id": 1, "name": "Center Back"},
         "jersey_number": i + 1} for i in range(11)]})


def _full_match_events() -> pd.DataFrame:
    """The dominance match with lineups, clock times and a half end in each period."""
    rows = [_event("Starting XI", A, tactics=_lineup(10)), _event("Starting XI", B, tactics=_lineup(30))]
    rows += _match_events().drop(columns="id").to_dict("records")
    rows += [_event("Half End", A, x=np.nan), _event("Half End", A, x=np.nan, period=2)]
    ev = pd.DataFrame(rows)
    ev["index"] = range(1, len(ev) + 1)
    ev["id"] = [f"e{i}" for i in range(len(ev))]
    ev["period_seconds"] = np.where(ev["type"] == "Half End", 2700.0, ev["index"] * 10.0)
    ev["minute"] = (ev["period_seconds"] // 60).astype(int) + np.where(ev["period"] == 2, 45, 0)
    ev["possession"] = np.where(ev["team_id"] == A, 1, 2)
    return ev


def _provider_table(team_match: pd.DataFrame) -> pd.DataFrame:
    keys = team_match[tables.PROVIDER_KEYS].copy()
    return keys.assign(**({c: 1.0 for c in tables.PROVIDER_COLUMNS} | {"hsr_without_ball_km": [2.0, 3.0]}))


def test_build_team_match_writes_every_new_column() -> None:
    tm, minutes = tables.build_team_match(_full_match_events(), _matches())
    assert len(tm) == 2
    assert set(tables.NEW_COLUMNS) <= set(tm.columns)
    assert tm[list(tables.PROVIDER_COLUMNS) + ["hsr_without_ball_per_dominance"]].isna().all().all()
    assert tm["players_used"].tolist() == [11, 11]
    assert tm["possession_share"].sum() == pytest.approx(1.0)
    assert len(minutes) == 22 and minutes["started"].all()


def test_team_match_provider_merge_present() -> None:
    tm, _ = tables.build_team_match(_full_match_events(), _matches())
    merged, _ = tables.build_team_match(_full_match_events(), _matches(), provider_table=_provider_table(tm))
    merged = merged.set_index("team")
    assert merged.loc["Alpha", "hsr_without_ball_km"] == 2.0
    assert merged.loc["Alpha", "distance_km"] == 1.0
    expected = 2.0 * 1000 / (merged.loc["Alpha", "dominance"] * 100)
    assert merged.loc["Alpha", "hsr_without_ball_per_dominance"] == pytest.approx(expected)


def test_team_match_provider_merge_absent_keeps_columns_empty() -> None:
    tm = pd.DataFrame({"tournament": "women", "match_id": 1, "team": ["Alpha", "Beta"],
                       "dominance": [0.6, 0.4]})
    out = tables.merge_provider(tm, None)
    assert set(tables.PROVIDER_COLUMNS) <= set(out.columns)
    assert out[list(tables.PROVIDER_COLUMNS) + ["hsr_without_ball_per_dominance"]].isna().all().all()


def test_team_match_provider_merge_leaves_women_empty() -> None:
    tm = pd.DataFrame({"tournament": ["men", "women"], "match_id": [1, 2], "team": ["Alpha", "Alpha"],
                       "dominance": [0.5, 0.5]})
    provider_table = pd.DataFrame({"tournament": ["men"], "match_id": [1], "team": ["Alpha"],
                        "hsr_without_ball_km": [4.0]})
    out = tables.merge_provider(tm, provider_table).set_index("tournament")
    assert out.loc["men", "hsr_without_ball_per_dominance"] == pytest.approx(80.0)
    assert np.isnan(out.loc["women", "hsr_without_ball_km"])


def test_team_tournament_means_per_match() -> None:
    tm = pd.DataFrame({"tournament": "men", "match_id": [1, 2], "team": "Alpha", "team_id": A,
                       "opponent_id": [B, B], "np_xg": [1.0, 2.0], "dominance": [0.4, 0.6],
                       "chances_both_zero": [False, True]})
    out = tables.team_tournament(tm).iloc[0]
    assert out["matches"] == 2
    assert out["np_xg"] == pytest.approx(1.5) and out["dominance"] == pytest.approx(0.5)
    assert "chances_both_zero" not in out.index and "opponent_id" not in out.index


def test_tracking_columns_empty_for_excluded_matches() -> None:
    tm, _ = tables.build_team_match(_full_match_events(), _matches())
    provider_table = _provider_table(tm)
    excluded = tm[["tournament", "match_id"]].drop_duplicates().assign(filled_in_share=0.7)
    merged, _ = tables.build_team_match(_full_match_events(), _matches(), provider_table=provider_table, excluded=excluded)
    assert merged[list(tables.TRACKING_COLUMNS) + ["hsr_without_ball_per_dominance"]].isna().all().all()
    kept = [c for c in tables.PROVIDER_COLUMNS if c not in tables.TRACKING_COLUMNS]
    assert (merged[kept] == 1.0).all().all()
    expected = tables.provider_expected(merged, provider_table, excluded)
    assert not expected[list(tables.TRACKING_COLUMNS)].any().any() and expected[kept].all().all()
    rows: list = []
    for col in tables.PROVIDER_COLUMNS:
        tables._coverage_check(rows, "men", merged.assign(tournament="men"), col, expected)
    assert all(r["passed"] for r in rows)
