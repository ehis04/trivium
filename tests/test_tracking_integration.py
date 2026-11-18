from __future__ import annotations

import pandas as pd
import pytest

from three_trees.features.efficiency import efficiency_position_group
from three_trees.tracking.running import position_group, with_position_group
from three_trees.tracking.stream import with_statsbomb_teams


@pytest.mark.parametrize(
    "position, category, expected",
    [("G", "player", "Goalkeeper"), ("GK", None, "Goalkeeper"), ("D", "player", "Defender"),
     ("DF", None, "Defender"), ("M", "player", "Midfielder"), ("Midfielder", None, "Midfielder"),
     ("F", "player", "Forward"), ("FW", None, "Forward"), (None, "goalkeeper", "Goalkeeper"),
     (None, "player", "Unknown"), ("X", "player", "Unknown")],
)
def test_position_group(position: object, category: object, expected: str) -> None:
    assert position_group(position, category) == expected


def _running() -> pd.DataFrame:
    rows = []
    for match_id, teams in ((1, ("KOREA REPUBLIC", "ITALY")), (2, ("ITALY", "Atlantis"))):
        for team in teams:
            for pid, pos in ((10, "D"), (11, "M")):
                rows.append({"tournament": "men", "match_id": match_id, "team": team, "person_id": pid,
                             "jersey_number": pid, "category_name": "player", "position": pos,
                             "state": "without_ball", "third": "defensive", "minutes": 45.0,
                             "distance_m": 4000.0, "hsr_m": 300.0, "sprint_m": 80.0})
    return pd.DataFrame(rows)


def test_running_teams_mapped_to_statsbomb_names_per_match() -> None:
    matches = pd.DataFrame({"match_id": [1, 2], "home_team": ["Korea Republic", "Italy"],
                            "away_team": ["Italy", "Spain"]})
    out = with_statsbomb_teams(with_position_group(_running()), matches)
    m1 = out[out["match_id"] == 1]
    assert set(m1["team"]) == {"Korea Republic", "Italy"}
    assert set(out["tracking_team"]) == {"KOREA REPUBLIC", "ITALY", "Atlantis"}
    # Match 2: Italy matches, so the other tracking name is the match's other team, Spain.
    assert set(out.loc[out["tracking_team"] == "Atlantis", "team"]) == {"Spain"}
    assert set(out["position_group"]) == {"Defender", "Midfielder"}


def test_running_teams_left_unmapped_when_neither_name_matches() -> None:
    matches = pd.DataFrame({"match_id": [2], "home_team": ["Mali"], "away_team": ["Spain"]})
    out = with_statsbomb_teams(with_position_group(_running()[lambda d: d["match_id"] == 2]), matches)
    assert out["team"].isna().all()


def test_mapped_running_joins_team_match_for_position_group_efficiency() -> None:
    matches = pd.DataFrame({"match_id": [1], "home_team": ["Korea Republic"], "away_team": ["Italy"]})
    running = with_statsbomb_teams(with_position_group(_running()[lambda d: d["match_id"] == 1]), matches)
    team_match = pd.DataFrame({"tournament": "men", "match_id": [1, 1], "team": ["Korea Republic", "Italy"],
                               "dominance": [0.55, 0.45]})
    out = efficiency_position_group(running, team_match)
    assert len(out) == 4
    assert set(out["team"]) == {"Korea Republic", "Italy"}


def test_team_match_provider_sums_fifa_and_tracking_per_team() -> None:
    """FIFA totals and tracking totals per team-match, in km; HSR split by state; top-3 share of HSR."""
    from three_trees.tracking.running import team_match_provider

    running = pd.DataFrame([
        {"tournament": "men", "match_id": 1, "team": "Italy", "person_id": p, "state": s, "third": t,
         "distance_m": 1000.0, "hsr_m": h}
        for p, h in ((1, 400.0), (2, 300.0), (3, 200.0), (4, 100.0))
        for s, t in (("with_ball", "middle"), ("without_ball", "defensive"))
    ])
    physical = pd.DataFrame({"tournament": "men", "match_id": [1, 1, 2], "team": ["Italy", "Italy", "Qatar"],
                             "total_distance_m": [4000.0, 4100.0, 9000.0], "dist_20_25_kmh_m": [300.0, 200.0, 0.0],
                             "dist_25_plus_kmh_m": [100.0, 50.0, 0.0]})
    out = team_match_provider(running, physical).set_index(["match_id", "team"])

    italy = out.loc[(1, "Italy")]
    assert italy["distance_km"] == pytest.approx(8.1) and italy["hsr_km"] == pytest.approx(0.65)
    assert italy["sprint_km"] == pytest.approx(0.15)
    assert italy["distance_km_tracking"] == pytest.approx(8.0) and italy["hsr_km_tracking"] == pytest.approx(2.0)
    assert italy["hsr_with_ball_km"] == pytest.approx(1.0) and italy["hsr_without_ball_km"] == pytest.approx(1.0)
    assert italy["hsr_out_of_play_km"] == 0 and italy["hsr_without_ball_defensive_third_km"] == pytest.approx(1.0)
    assert italy["top3_hsr_share"] == pytest.approx(0.9)  # (800 + 600 + 400) / 2000
    # Qatar has FIFA rows but no tracking: tracking columns stay empty.
    assert out.loc[(2, "Qatar"), "distance_km"] == pytest.approx(9.0)
    assert pd.isna(out.loc[(2, "Qatar"), "distance_km_tracking"])
